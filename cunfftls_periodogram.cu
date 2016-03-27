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
//#include <curand.h>
//#include <curand_kernel.h>

#include "cuna.h"
#include "cuna_filter.h"
#include "cunfftls_typedefs.h"
#include "cunfftls_utils.h"

#ifdef DOUBLE_PRECISION
#define FILTER_RADIUS 12
#else
#define FILTER_RADIUS 6
#endif

// CUDA kernel for converting spectral/window functions to LSP
__global__ void
convertToLSP( const Complex *sp, const Complex *win, const dTyp var, const int m, const int npts, const int nlsp, dTyp *lsp) {

  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int j = i % m;
  int off = (i - j)*2;
  if (j == m-1 && i < m * nlsp) lsp[i] = 0.;
  else if ( i < m * nlsp ) {
    Complex z1 = sp[ j + 1 + off ];
    Complex z2 = win[ 2 * (j + 1)];
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
    lsp[i] = (cterm + sterm) / ((npts - 1) * var);
  }
}


// CUDA kernel for converting spectral/window functions to GENERALIZED (i.e. floating mean) LSP
__global__ void
convertToGLSP( const Complex *sp, const Complex *win, const dTyp YY, const int m, const int npts, const int nlsp, dTyp *lsp) {

  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int j = i % m;
  int off = (i - j)*2;

  if (j == m-1 && i < m * nlsp) lsp[i] = 0.;
  else if ( i < m * nlsp ) {
    Complex z1 =  sp[ j + 1 + off ];
    Complex z2 = win[ 2 * (j + 1) + off];
    Complex z3 = win[ j + 1 + off];

    dTyp YS = cuImag(z1);
    dTyp YC = cuReal(z1);

    dTyp S2 = cuImag(z2);
    dTyp C2 = cuReal(z2);

    dTyp S  = cuImag(z3);
    dTyp C  = cuReal(z3);

    dTyp CChat = 0.5 * (1 + C2);
    
    dTyp SS    = 1 - CChat - S * S; 
    dTyp CC    = CChat - C * C;
    dTyp CS    = 0.5 * S2 - C * S;
    dTyp D     = CC * SS - CS * CS;

    lsp[i]     = (1./(YY * D)) 
                 * (SS * YC * YC 
                    + CC * YS * YS 
                    - 2 * CS * YC * YS);
   
  }
}

// ensures observed times are in [-1/2, 1/2)
__host__ void
scaleTobs(const dTyp *tobs, int npts, dTyp over, dTyp *t) {

  // now transform t -> [-1/2, 1/2)
  dTyp tmax     = tobs[npts - 1];
  dTyp tmin     = tobs[0];
  
  dTyp invrange = 1./((tmax - tmin) * over);
  dTyp a        = 0.5 - EPSILON;

  for(int i = 0; i < npts; i++) 
    t[i] = 2 * a * (tobs[i] - tmin) * invrange - a;
  
}


// subtracts mean from observations
__host__ void
scaleYobs(const dTyp *yobs, const int npts, dTyp *var, dTyp *y) {
  dTyp avg;
  meanAndVariance(npts, yobs, &avg, var);
  
  for(int i = 0; i < npts; i++)
    y[i] = yobs[i] - avg;
  
}

// subtracts mean from observations
__host__ void
scaleYobsWeighted(const dTyp *yobs, const dTyp *w, const int npts, dTyp *var, dTyp *Y) {
  dTyp avg;
  weightedMeanAndVariance(npts, yobs, w, &avg, var);
  
  for(int i = 0; i < npts; i++) 
    Y[i] = w[i] * (yobs[i] - avg);
  
}

// computes + returns pointer to periodogram
__host__  dTyp *
lombScargle(const dTyp *tobs, const dTyp *yobs, int npts, 
            Settings *settings) {

  int ng = settings->nfreqs * 2;
  int nbs = settings->nbootstraps;
  
  // calculate total host memory and ensure there's enough
  size_t total_host_mem =        2 * npts * sizeof(dTyp)
                     + (ng/2) * (nbs + 1) * sizeof(dTyp)
                     +                      sizeof(filter_properties);

  if (total_host_mem > settings->host_memory) {
        eprint("total host memory allocated is %d bytes, but %d "
                  "bytes are required to perform LSP", settings->host_memory, total_host_mem);
        exit(EXIT_FAILURE);
  }

  // setup pointers
  dTyp *t      = (dTyp *)  settings->host_workspace; // length npts
  dTyp *y      = (dTyp *)(t          +        npts); // length npts 
  dTyp *lsp    = (dTyp *)(y          +        npts); // length ng/2 * (nbs + 1)
  filter_properties *fprops = (filter_properties *)(lsp +  (ng/2) * (nbs + 1));
  
  // scale t and y (zero mean, t \in [-1/2, 1/2))
  dTyp var;
  scaleTobs(tobs, npts, settings->over, t);
  scaleYobs(yobs, npts, &var,           y);
 
  // calculate total device memory and ensure we have enough 
  size_t total_device_mem = total_host_mem 
                       +     ng * (nbs + 3) * sizeof(Complex)
                       +           2 * npts * sizeof(dTyp)
                       +      FILTER_RADIUS * sizeof(dTyp);
 
  if (total_device_mem > settings->device_memory) {
        eprint("total device memory allocated is %d bytes, but %d "
                  "bytes are required to perform LSP", settings->device_memory, total_device_mem);
        exit(EXIT_FAILURE);
  }
  
  // setup pointers
  dTyp *d_t            = (dTyp *)settings->device_workspace; // length npts
  dTyp *d_y            = (dTyp *)   (d_t          +   npts); // length npts
  dTyp *d_lsp          = (dTyp *)   (d_y          +   npts); // length ng/2 *(nbs + 1)
  filter_properties *d_fprops       = (filter_properties *)(d_lsp + (ng/2) * (nbs + 1));

  fprops->E1           = (dTyp *)   (d_fprops     +      1); // length npts
  fprops->E2           = (dTyp *)   (fprops->E1   +   npts); // length npts
  fprops->E3           = (dTyp *)   (fprops->E2   +   npts); // length FILTER_RADIUS  
  Complex *d_f_hat     = (Complex *)(fprops->E3   + FILTER_RADIUS); // length ng  * (nbs + 1)
  Complex *d_f_hat_win = (Complex *)(d_f_hat      +     ng * (nbs + 1)); // length 2 * ng
  
  // transfer data to gpu
  checkCudaErrors(cudaMemcpyAsync(settings->device_workspace, settings->host_workspace,
        total_host_mem, cudaMemcpyHostToDevice, settings->stream));
   
  // checkCudaErrors(cudaStreamSynchronize(settings->stream));

  // generate filter
  generate_pinned_filter_properties(d_t, npts, ng, fprops, d_fprops, settings->stream); 
  // checkCudaErrors(cudaGetLastError());
  // checkCudaErrors(cudaStreamSynchronize(settings->stream));

  // memset
  checkCudaErrors(cudaMemsetAsync(d_f_hat, 0,  ng * (nbs + 3) * sizeof(Complex), settings->stream));
  // checkCudaErrors(cudaStreamSynchronize(settings->stream));

  // evaluate NFFT for signal & window
  cunfft_adjoint_raw_async(d_t,  d_y, d_f_hat,     npts,     ng, d_fprops, settings->stream);
  cunfft_adjoint_raw_async(d_t, NULL, d_f_hat_win, npts, 2 * ng, d_fprops, settings->stream); 
  if (nbs > 0) {
    cunfft_adjoint_raw_async_bootstrap(d_t,  d_y, d_f_hat + ng,     npts,     ng, nbs, d_fprops, settings->stream);
    //cunfft_adjoint_raw_async_bootstrap(d_t, NULL, d_f_hat_win + 2 * ng, npts, 2 * ng, nbs, d_fprops, settings->stream); 
    //checkCudaErrors(cudaStreamSynchronize(settings->stream));
  }
  // calculate number of CUDA blocks we need
  size_t nblocks = (ng/2) * (nbs + 1)/ BLOCK_SIZE;
  while (nblocks * BLOCK_SIZE < (ng/2) * (nbs + 1)) nblocks++;
  
  // convert to LSP
  convertToLSP <<< nblocks, BLOCK_SIZE, 0, settings->stream >>> 
           (d_f_hat, d_f_hat_win, var, ng/2, npts, nbs + 1, d_lsp);
//  checkCudaErrors(cudaGetLastError());
//  checkCudaErrors(cudaStreamSynchronize(settings->stream));

  // Copy the results back to CPU memory
  checkCudaErrors(cudaMemcpyAsync(lsp, d_lsp, (ng/2) * (nbs + 1) * sizeof(dTyp), 
        cudaMemcpyDeviceToHost, settings->stream));
//  checkCudaErrors(cudaStreamSynchronize(settings->stream));

  return lsp;
}

__host__ void convertErrorsToWeights(const dTyp *errs, dTyp *weights, const int n) {
  
  dTyp W = 0;
  if (errs == NULL) {
     W = 1./n;
     for(int i = 0; i < n; i++) 
        weights[i] = W;
     return;
  }
  for(int i = 0; i < n; i++) {
    weights[i] = 1./(errs[i] * errs[i]);
    W         += weights[i];
  }
  dTyp Winv = 1./W;
  for(int i = 0; i < n; i++)
    weights[i] *= Winv;
}

// computes + returns pointer to periodogram
__host__  dTyp *
generalizedLombScargle(const dTyp *tobs, const dTyp *yobs, const dTyp *errs, int npts, 
            Settings *settings) {

  int ng = settings->nfreqs * 2;
  int nbs = settings->nbootstraps;
  
  // calculate total host memory and ensure there's enough
  size_t total_host_mem =          3 * npts * sizeof(dTyp)
                     +   (ng/2) * (nbs + 1) * sizeof(dTyp)
                     +                        sizeof(filter_properties);

  if (total_host_mem > settings->host_memory) {
        eprint("total host memory is %d bytes, but we need %d "
                  "bytes to perform LSP", settings->host_memory, total_host_mem);
        exit(EXIT_FAILURE);
  }

  // setup (host) pointers
  dTyp *t      = (dTyp *)  settings->host_workspace; // length npts
  dTyp *y      = (dTyp *)(t          +        npts); // length npts 
  dTyp *w      = (dTyp *)(y          +        npts); // length npts
  dTyp *lsp    = (dTyp *)(w          +        npts); // length ng/2 * (nbs + 1)
  filter_properties *fprops = (filter_properties *)(lsp + (ng/2) * (nbs + 1));
  
  // scale t and y (zero weighted mean, t \in [-1/2, 1/2))
  dTyp var;
  convertErrorsToWeights(errs, w, npts);
  scaleTobs(tobs, npts, settings->over,  t);
  scaleYobsWeighted(yobs, w, npts, &var, y);
 
  // calculate total device memory and ensure we have enough 
  size_t total_device_mem = total_host_mem 
                       + 3 * (nbs + 1) * ng * sizeof(Complex)
                       +           2 * npts * sizeof(dTyp)
                       +      FILTER_RADIUS * sizeof(dTyp);
 
  if (total_device_mem > settings->device_memory) {
        eprint("total device memory is %d bytes, but we need %d "
                  "bytes to perform LSP", settings->device_memory, total_device_mem);
        exit(EXIT_FAILURE);
  }
  
  // setup (device) pointers
  dTyp *d_t            = (dTyp *)settings->device_workspace; // length npts
  dTyp *d_y            = (dTyp *)   (d_t          +   npts); // length npts
  dTyp *d_w            = (dTyp *)   (d_y          +   npts); // length npts
  dTyp *d_lsp          = (dTyp *)   (d_w          +   npts); // length ng/2 * (nbs + 1)
  filter_properties *d_fprops       = (filter_properties *)(d_lsp + (ng/2) * (nbs + 1));

  fprops->E1           = (dTyp *)   (d_fprops     +      1); // length npts
  fprops->E2           = (dTyp *)   (fprops->E1   +   npts); // length npts
  fprops->E3           = (dTyp *)   (fprops->E2   +   npts); // length FILTER_RADIUS  
  Complex *d_f_hat     = (Complex *)(fprops->E3   + FILTER_RADIUS); // length ng * (nbs + 1) 
  Complex *d_f_hat_win = (Complex *)(d_f_hat      + (nbs + 1) * ng); // length 2 * ng * (nbs + 1)
  
  // transfer data to gpu
  checkCudaErrors(cudaMemcpyAsync(settings->device_workspace, settings->host_workspace,
        total_host_mem, cudaMemcpyHostToDevice, settings->stream));
  //checkCudaErrors(cudaStreamSynchronize(settings->stream));

  // generate filter
  generate_pinned_filter_properties(d_t, npts, ng, fprops, d_fprops, settings->stream); 
  //checkCudaErrors(cudaStreamSynchronize(settings->stream));

  // memset
  checkCudaErrors(cudaMemsetAsync(d_f_hat, 0, 3 * ng * (nbs + 1) * sizeof(Complex), settings->stream));
  //checkCudaErrors(cudaStreamSynchronize(settings->stream));

  // evaluate NFFT for signal & window
  cunfft_adjoint_raw_async(d_t,  d_y, d_f_hat,     npts,     ng, d_fprops, settings->stream);
  cunfft_adjoint_raw_async(d_t,  d_w, d_f_hat_win, npts, 2 * ng, d_fprops, settings->stream); 
  //checkCudaErrors(cudaStreamSynchronize(settings->stream));
  if (nbs > 0) {
    // TODO: I think it's a problem that the weights and the datapoints are shuffled independently.
    //       i.e. instead of shuffle([ (y, y_w), ... ]) we're doing shuffle([ y, y2, ...]) AND
    //       shuffle([ y_w, y_w2, ... ]). Obviously not a problem for constant weights, but things
    //       get hairy when that's not the case.
    cunfft_adjoint_raw_async_bootstrap(d_t, d_y, d_f_hat     + ng    , npts, ng    , 
				nbs, d_fprops, settings->stream);
    cunfft_adjoint_raw_async_bootstrap(d_t, d_w, d_f_hat_win + 2 * ng, npts, 2 * ng, 
    				nbs, d_fprops, settings->stream);
    //checkCudaErrors(cudaStreamSynchronize(settings->stream));
  }

  // calculate number of CUDA blocks we need
  size_t nblocks = (ng/2) * (nbs + 1) / BLOCK_SIZE;
  while (nblocks * BLOCK_SIZE < (ng/2) * (nbs + 1)) nblocks++;
  
  // convert to (generalized) LSP
  convertToGLSP <<< nblocks, BLOCK_SIZE, 0, settings->stream >>> 
           (d_f_hat, d_f_hat_win, var, ng/2, npts, nbs + 1, d_lsp);
  checkCudaErrors(cudaGetLastError());
  //checkCudaErrors(cudaStreamSynchronize(settings->stream));

  // Copy the results back to CPU memory
  checkCudaErrors(cudaMemcpyAsync(lsp, d_lsp, (ng/2) * (nbs + 1) * sizeof(dTyp), 
        cudaMemcpyDeviceToHost, settings->stream));

  //checkCudaErrors(cudaStreamSynchronize(settings->stream));
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
  dTyp Ix = 1.0 - pow(1 - 2 * (npts - 1) * Pn / npts, 0.5 * (npts - 3));

  dTyp proba = 1 - pow(Ix, effm);
  if (proba < EPSILON) 
	proba = effm * pow( 1 - 2 * (npts - 1) * Pn / npts, 0.5 * (npts - 3));
  
  return proba;
}

__host__ dTyp logProba(dTyp Pn, int npts, int nfreqs, dTyp over) {
  dTyp proba = probability(Pn, npts, nfreqs, over);
  
  if (proba < EPSILON)
	return log10(2) + log10(nfreqs) - log10(over) 
		+ 0.5 * (npts - 3) 
		* (log10(npts - 2 * (npts - 1) * Pn) - log10(npts));
  else
	return log10(proba);
}
