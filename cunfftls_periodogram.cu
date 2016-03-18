/* cunfftls_periodogram.cu
 * =======================
 * CUDA implementation of the Lomb-Scargle periodogram
 * 
 * depends on the cunfft_adjoint library
 * 
 * (c) 2016, John Hoffman 
 * code borrowed extensively from B. Leroy's nfftls
 *
 * This file is part of cunfftls.
 *
 * cunfftls is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cunfftls is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cunfftls.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Copyright (C) 2016 J. Hoffman, 2012 B. Leroy [nfftls]
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cuComplex.h>

#include "cuna.h"
#include "cuna_filter.h"
#include "cunfftls_typedefs.h"
#include "cunfftls_utils.h"


// CUDA kernel for converting spectral/window functions to LSP
__global__ void
convertToLSP( const Complex *sp, const Complex *win, dTyp var, int m, int npts, dTyp *lsp) {

  int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (j == m-1) lsp[j] = 0.;
  else if ( j + 1 < m ) {
    Complex z1 = sp[ j + 1 ];
    Complex z2 = win[ 2 * (j + 1) ];
    dTyp invhypo = 1./cuAbs(z2);
    dTyp Sy = cuImag(z1);
    dTyp S2 = cuImag(z2);
    dTyp Cy = cuReal(z1);
    dTyp C2 = cuReal(z2);
    dTyp hc2wtau = 0.5 * C2 * invhypo;
    dTyp hs2wtau = 0.5 * S2 * invhypo;
    dTyp cwtau = cuSqrt( 0.5 + hc2wtau);
    dTyp swtau = cuSqrt( 0.5 - hc2wtau);
    if( S2 < 0 ) swtau *= -1;
    dTyp ycoswt_tau = Cy * cwtau + Sy * swtau;
    dTyp ysinwt_tau = Sy * cwtau - Cy * swtau;
    dTyp sum = hc2wtau * C2 + hs2wtau * S2;
    dTyp cos2wttau = 0.5 * m + sum;
    dTyp sin2wttau = 0.5 * m - sum;
    dTyp cterm = square(ycoswt_tau) / cos2wttau;
    dTyp sterm = square(ysinwt_tau) / sin2wttau;
    lsp[j] = (cterm + sterm) / (2 * var);
    /*
    dTyp c2ttau = 0.5 * npts + 0.5 * C2 * (C2 * invhypo) + 0.5 * Sy 
    dTyp hc2wt = 0.5 * cuReal(z2)  * invhypo;
    dTyp hs2wt = 0.5 * cuImag(z2)  * invhypo;
    dTyp cwt = cuSqrt(0.5 + hc2wt);
    dTyp swt = sign(cuSqrt(0.5 - hc2wt), hs2wt);
    dTyp den = 0.5 * npts + hc2wt * cuReal(z2) + hs2wt * cuImag(z2);
    dTyp cterm = square(cwt * cuReal(z1) + swt * cuImag(z1)) / den;
    dTyp sterm = square(cwt * cuImag(z1) - swt * cuReal(z1)) / (npts - den);
    
    lsp[j] = (cterm + sterm) / (2 * var);*/
  }
}

// ensures observed times are in [-1/2, 1/2)
__host__ dTyp *
scaleTobs(const dTyp *tobs, int npts, dTyp over) {

  // clone data
  dTyp * t = (dTyp *)malloc(npts * sizeof(dTyp));

  // now transform t -> [-1/2, 1/2)
  dTyp tmax     = tobs[npts - 1];
  dTyp tmin     = tobs[0];
  
  dTyp invrange = 1./((tmax - tmin) * over);
  dTyp a        = 0.5 - EPSILON;

  for(int i = 0; i < npts; i++) 
    t[i] = 2 * a * (tobs[i] - tmin) * invrange - a;
  
  return t;
}


// subtracts mean from observations
__host__ dTyp *
scaleYobs(const dTyp *yobs, int npts, dTyp *var) {
  dTyp avg;
  meanAndVariance(npts, yobs, &avg, var);
  
  dTyp *y = (dTyp *)malloc(npts * sizeof(dTyp));
  for(int i = 0; i < npts; i++)
    y[i] = yobs[i] - avg;
  
  return y;
}

// computes + returns pointer to periodogram
__host__  dTyp *
lombScargle(const dTyp *tobs, const dTyp *yobs, int npts, 
            Settings *settings) {

  clock_t start;
  int ng = settings->nfreqs * 2;

  START_TIMER;
  // scale t and y (zero mean, t \in [-1/2, 1/2))
  dTyp var;
  dTyp *t = scaleTobs(tobs, npts, settings->over);
  dTyp *y = scaleYobs(yobs, npts, &var);
  STOP_TIMER("scale x and y axes", start);
  
  dTyp *lsp = (dTyp *) malloc((ng/2) * sizeof(dTyp));


  START_TIMER;
  // declare 
  dTyp *d_t, *d_y, *d_ygrid, *d_lsp;
  Complex *d_f_hat, *d_f_hat_win;

  // allocate
  checkCudaErrors(cudaMalloc((void **) &d_t,           npts * sizeof(dTyp)));
  checkCudaErrors(cudaMalloc((void **) &d_y,           npts * sizeof(dTyp)));
  checkCudaErrors(cudaMalloc((void **) &d_ygrid,     2 * ng * sizeof(dTyp))); // shared with window
  checkCudaErrors(cudaMalloc((void **) &d_lsp,       (ng/2) * sizeof(dTyp)));
  checkCudaErrors(cudaMalloc((void **) &d_f_hat,         ng * sizeof(Complex)));
  checkCudaErrors(cudaMalloc((void **) &d_f_hat_win, 2 * ng * sizeof(Complex)));

  // memset
  checkCudaErrors(cudaMemset(d_ygrid,       0, 2 * ng * sizeof(dTyp)));
  //checkCudaErrors(cudaMemset(d_lsp,         0, (ng/2) * sizeof(dTyp)));
  //checkCudaErrors(cudaMemset(d_f_hat,       0,     ng * sizeof(Complex)));
  //checkCudaErrors(cudaMemset(d_f_hat_win,   0, 2 * ng * sizeof(Complex)));
  STOP_TIMER("cuda malloc all gpu vars", start);
  
  
  // transfer data to gpu
  START_TIMER;
  checkCudaErrors(cudaMemcpy(d_t, t, npts * sizeof(dTyp), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, y, npts * sizeof(dTyp), cudaMemcpyHostToDevice));
  STOP_TIMER("transfer data to device", start);

  // generate filter
  START_TIMER;
  filter_properties *fprops_host, *fprops_device;
  generate_filter_properties(d_t, npts, ng, &fprops_host, &fprops_device); 
  STOP_TIMER("generate filter properties", start);

  // evaluate NFFT for signal & window
  START_TIMER;
  cunfft_adjoint_raw(d_t, d_y,  d_ygrid, d_f_hat,     npts,     ng, fprops_device);
  checkCudaErrors(cudaMemset(d_ygrid,       0, ng * sizeof(dTyp)));
  cunfft_adjoint_raw(d_t, NULL, d_ygrid, d_f_hat_win, npts, 2 * ng, fprops_device); 
  STOP_TIMER("cunfft_adjoint_raw", start);

  // calculate number of CUDA blocks we need
  int nblocks = (ng/2) / BLOCK_SIZE;
  while (nblocks * BLOCK_SIZE < (ng/2)) nblocks++;
  
  // convert to LSP
  START_TIMER;
  convertToLSP <<< nblocks, BLOCK_SIZE >>> 
           (d_f_hat, d_f_hat_win, var, ng/2, npts, d_lsp);
  STOP_TIMER("convertToLSP", start);

  // Copy the results back to CPU memory
  START_TIMER;
  checkCudaErrors(cudaMemcpy(lsp, d_lsp, (ng/2) * sizeof(dTyp), cudaMemcpyDeviceToHost));
  STOP_TIMER("malloc + memcpy lsp", start);

  // Free memory
  free(t);
  free(y);
  checkCudaErrors(cudaFree(d_lsp));
  checkCudaErrors(cudaFree(d_ygrid));
  checkCudaErrors(cudaFree(d_y));
  checkCudaErrors(cudaFree(d_t));
  checkCudaErrors(cudaFree(d_f_hat_win));
  checkCudaErrors(cudaFree(d_f_hat));

  return lsp;

}

// below is taken directly from nfftls (B. Leroy)
/**
 * Returns the probability that a peak of a given power
 * appears in the periodogram when the signal is white
 * Gaussian noise.
 *
 * \param Pn the power in the periodogram.
 * \param npts the number of samples.
 * \param nfreqs the number of frequencies in the periodogram.
 * \param over the oversampling factor.
 *
 * \note This is the expression proposed by A. Schwarzenberg-Czerny
 * (MNRAS 1998, 301, 831), but without explicitely using modified
 * Bessel functions.
 * \note The number of effective independent frequencies, effm,
 * is the rough estimate suggested in Numerical Recipes. 
 */
__host__
dTyp 
probability(dTyp Pn, int npts, int nfreqs, dTyp over)
{
  dTyp effm = 2.0 * nfreqs / over;
  dTyp Ix = 1.0 - pow(1 - 2 * Pn / npts, 0.5 * (npts - 3));

  dTyp proba = 1 - pow(Ix, effm);
  if (proba < EPSILON) 
	proba = effm * pow( 1 - 2 * Pn / npts, 0.5 * (npts - 3));
  
  return proba;
}

__host__ dTyp logProba(dTyp Pn, int npts, int nfreqs, dTyp over) {
  dTyp proba = probability(Pn, npts, nfreqs, over);
  
  if (proba < EPSILON)
	return log10(2) + log10(nfreqs) - log10(over) + 0.5 * (npts - 3) * (log10(npts - 2 * Pn) - log10(npts));
  else
	return log10(proba);
}
