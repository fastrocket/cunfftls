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
#include <time.h>

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
    //lsp[i] = (cterm + sterm) / ((npts - 1) * var);
    lsp[i] = (cterm + sterm) / var;
  }
}


// CUDA kernel for converting spectral/window functions to GENERALIZED (i.e. floating mean) LSP
__global__ void
convertToGLSP( const Complex *sp, const Complex *win, const dTyp YY, const int m, const int npts, const int nlsp, dTyp *lsp) {
  // TODO: Fix this. Numerically unstable? Will produce negative values, etc.
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

    dTyp CChat = 0.5 * (1 + C2); //+
    dTyp SShat = 0.5 * (1 - C2); //+

    dTyp SS    = SShat - S * S;     //+
    dTyp CC    = CChat - C * C;     //+
    dTyp CS    = 0.5 * S2 - C * S;  //+
    dTyp D     = CC * SS - CS * CS; //+

    lsp[i]     = (SS * YC * YC 
                    + CC * YS * YS 
                    - 2 * CS * YC * YS)/(YY * D);
    /*if (lsp[i] < 0 ) {
     printf("i = %d  j = %d"
      "\n\tSS = %e       CC = %e         CS = %e       D = %e"
      "\n\tS = %e        C = %e          S2 = %e       C2 = %e"
      "\n\tYS = %e       YC = %e         YY*D = %e"
      "\n\tSS * YC * YC = %e    CC * YS * YS = %e     -2 * CS * YC * YS = %e"
      "\n\tsum of first 2 terms = %e       3 terms = %e\n",i, j, SS, CC, CS, D, S, C, S2, C2, YS, YC, YY *D,
        SS*YC*YC, CC*YS*YS, -2 * CS * YC * YS, 
        SS * YC * YC + CC * YS * YS,SS * YC * YC + CC * YS * YS - 2 * CS * YC * YS );
     }*/
    /*
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
    lsp[i] = (cterm + sterm) / ((npts - 1) * YY);
    */
  }
}


// CUDA kernel for converting spectral/window functions to GENERALIZED (i.e. floating mean) LSP
__global__ void
convertToGLSP_usingTimeShift( const Complex *sp, const Complex *win, const dTyp YY, const int m, const int npts, const int nlsp, dTyp *lsp) {
  // TODO: Fix this. Numerically unstable? Will produce negative values, etc.
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

    dTyp CChat = 0.5 * (1 + C2); //+
    dTyp SShat = 0.5 * (1 - C2); //+

    dTyp SS    = SShat - S * S;     //+
    dTyp CC    = CChat - C * C;     //+
    dTyp CS    = 0.5 * S2 - C * S;  //+
    //dTyp D     = CC * SS - CS * CS; //+

    //lsp[i]     = (SS * YC * YC 
      //              + CC * YS * YS 
        //            - 2 * CS * YC * YS)/(YY * D);

    dTyp tan2wt = (2 * CS) / (CC - SS);

    dTyp cos2wt  = 1.0 / cuSqrt(1 + square(tan2wt));
    dTyp sin2wt  = tan2wt * cos2wt;
    

    dTyp coswt = cuSqrt(0.5 * ( 1 + cos2wt ));
    dTyp sinwt = cuSqrt(0.5 * ( 1 - cos2wt ));

    if (tan2wt < 0) { 
        sin2wt *= -1;
        sinwt  *= -1;
    }

    dTyp YCtau = YC * coswt + YS * sinwt;
    dTyp YStau = YS * coswt - YC * sinwt;

    dTyp Ctau  = C * coswt + S * sinwt;
    dTyp Stau  = S * coswt - C * sinwt;

    dTyp C2tau = C2 * cos2wt + S2 * sin2wt;


    dTyp CCtau = 0.5 * ( 1 + C2tau ) - square(Ctau);
    dTyp SStau = 0.5 * ( 1 - C2tau ) - square(Stau);

    lsp[i] = (square(YCtau) / CCtau + square(YStau) / SStau) / YY ;
    /*if (lsp[i] < 0 ) {
     printf("i = %d  j = %d"
      "\n\tSS = %e       CC = %e         CS = %e       D = %e"
      "\n\tS = %e        C = %e          S2 = %e       C2 = %e"
      "\n\tYS = %e       YC = %e         YY*D = %e"
      "\n\tSS * YC * YC = %e    CC * YS * YS = %e     -2 * CS * YC * YS = %e"
      "\n\tsum of first 2 terms = %e       3 terms = %e\n",i, j, SS, CC, CS, D, S, C, S2, C2, YS, YC, YY *D,
        SS*YC*YC, CC*YS*YS, -2 * CS * YC * YS, 
        SS * YC * YC + CC * YS * YS,SS * YC * YC + CC * YS * YS - 2 * CS * YC * YS );
     }*/
    /*
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
    lsp[i] = (cterm + sterm) / ((npts - 1) * YY);
    */
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
  
  // for alignment purposes (GPU addresses must start at multiples of 8bytes)
  int lsp_pad = ((ng/2) * (nbs + 1)) % 2 == 0 ? 0 : 1;
  int lc_pad  =                 npts % 2 == 0 ? 0 : 1;
  
  // calculate total host memory and ensure there's enough
  size_t total_host_mem =         2 * (npts + lc_pad) * sizeof(dTyp)  // lc (x, y)
                     + ((ng/2) * (nbs + 1) + lsp_pad) * sizeof(dTyp)  // lsp
                     +                      sizeof(filter_properties);

  if (total_host_mem > settings->host_memory) {
        eprint("total host memory allocated is %d bytes, but %d "
                  "bytes are required to perform LSP", settings->host_memory, total_host_mem);
        exit(EXIT_FAILURE);
  }

 

  // setup pointers
  dTyp *t      = (dTyp *)  settings->host_workspace; // length npts
  dTyp *y      = (dTyp *)(t          +        npts + lc_pad); // length npts 
  dTyp *lsp    = (dTyp *)(y          +        npts + lc_pad); // length ng/2 * (nbs + 1)
  filter_properties *fprops = (filter_properties *)(lsp +  (ng/2) * (nbs + 1) + lsp_pad);
  
  // scale t and y (zero mean, t \in [-1/2, 1/2))
  dTyp var;
  scaleTobs(tobs, npts, settings->over, t);
  scaleYobs(yobs, npts, &var,           y);
 
  // calculate total device memory and ensure we have enough 
  size_t total_device_mem = total_host_mem 
                       +    ng * (nbs + 3) * sizeof(Complex)  //  d_f_hat (ng (nbs + 1)), d_f_hat_win (2ng)
                       +     2 * (npts + lc_pad) * sizeof(dTyp) // E1, E2
                       +   (2 * (FILTER_RADIUS  + 1)) * sizeof(dTyp); // E3
 
  if (total_device_mem > settings->device_memory) {
        eprint("total device memory allocated is %d bytes, but %d "
                  "bytes are required to perform LSP", settings->device_memory, total_device_mem);
        exit(EXIT_FAILURE);
  }
  
  // setup pointers
  dTyp *d_t            = (dTyp *)    settings->device_workspace; // length npts
  dTyp *d_y            = (dTyp *)   (d_t          +   npts + lc_pad); // length npts
  dTyp *d_lsp          = (dTyp *)   (d_y          +   npts + lc_pad); // length ng/2 *(nbs + 1)
  filter_properties *d_fprops    = (filter_properties *)(d_lsp + (ng/2) * (nbs + 1) + lsp_pad);

  fprops->E1           = (dTyp *)   (d_fprops     +   1); // length npts
  fprops->E2           = (dTyp *)   (fprops->E1   +   npts + lc_pad); // length npts
  fprops->E3           = (dTyp *)   (fprops->E2   +   npts + lc_pad); // length 2 * FILTER_RADIUS + 1 + 1 (alignment padding)
  Complex *d_f_hat     = (Complex *)(fprops->E3   +   2 * (FILTER_RADIUS + 1)); // length ng * (nbs + 1)
  Complex *d_f_hat_win = (Complex *)(d_f_hat      +  ng * (nbs + 1)); // length 2 * ng
  
  // transfer data to gpu
  checkCudaErrors(cudaMemcpyAsync(settings->device_workspace, settings->host_workspace,
        total_host_mem, cudaMemcpyHostToDevice, settings->stream));

  checkCudaErrors(cudaStreamSynchronize(settings->stream));
 
  // generate filter
  generate_pinned_filter_properties(d_t, npts, ng, fprops, d_fprops, settings->stream); 

  // memset
  checkCudaErrors(cudaMemsetAsync(d_f_hat, 0,  ng * (nbs + 3) * sizeof(Complex), settings->stream));
  // checkCudaErrors(cudaStreamSynchronize(settings->stream));

  // evaluate NFFT for signal & window
  cunfft_adjoint_raw_async(d_t,  d_y, d_f_hat,     npts,     ng, fprops, d_fprops, settings->stream);
  cunfft_adjoint_raw_async(d_t, NULL, d_f_hat_win, npts, 2 * ng, fprops, d_fprops, settings->stream); 
  if (nbs > 0) {
    cunfft_adjoint_raw_async_bootstrap(d_t,  d_y, d_f_hat + ng,     npts,     ng, nbs, 
						fprops, d_fprops, settings->stream, (unsigned int) clock());
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
  unsigned int seed;

  // for alignment purposes (GPU addresses must start at multiples of 8bytes)
  int lsp_pad = ((ng/2) * (nbs + 1)) % 2 == 0 ? 0 : 1;
  int lc_pad  =                 npts % 2 == 0 ? 0 : 1;
  
  // calculate total host memory and ensure there's enough
  size_t total_host_mem =          3 * (npts + lc_pad) * sizeof(dTyp)
                     +  ((ng/2) * (nbs + 1) + lsp_pad) * sizeof(dTyp)
                     +                        sizeof(filter_properties);

  if (total_host_mem > settings->host_memory) {
        eprint("total host memory is %d bytes, but we need %d "
                  "bytes to perform LSP", settings->host_memory, total_host_mem);
        exit(EXIT_FAILURE);
  }

  // setup (host) pointers
  dTyp *t      = (dTyp *)  settings->host_workspace; // length npts
  dTyp *y      = (dTyp *)(t          +        npts + lc_pad); // length npts 
  dTyp *w      = (dTyp *)(y          +        npts + lc_pad); // length npts
  dTyp *lsp    = (dTyp *)(w          +        npts + lc_pad); // length ng/2 * (nbs + 1)
  filter_properties *fprops = (filter_properties *)(lsp + (ng/2) * (nbs + 1) + lsp_pad);
  
  // scale t and y (zero weighted mean, t \in [-1/2, 1/2))
  dTyp var;
  convertErrorsToWeights(errs, w, npts);
  scaleTobs(tobs, npts, settings->over,  t);
  scaleYobsWeighted(yobs, w, npts, &var, y);
  //scaleYobs(yobs, npts, &var,           y);
  //for (int i = 0; i < npts; i++) w[i] = 1;
 
  // calculate total device memory and ensure we have enough 
  size_t total_device_mem = total_host_mem 
                       + 3 * (nbs + 1) * ng * sizeof(Complex)
                       +           2 * (npts + lc_pad) * sizeof(dTyp)
                       +      (2 * (FILTER_RADIUS + 1)) * sizeof(dTyp);
 
  if (total_device_mem > settings->device_memory) {
        eprint("total device memory is %d bytes, but we need %d "
                  "bytes to perform LSP", settings->device_memory, total_device_mem);
        exit(EXIT_FAILURE);
  }
  
  // setup (device) pointers
  dTyp *d_t            = (dTyp *)settings->device_workspace; // length npts
  dTyp *d_y            = (dTyp *)   (d_t          +   npts + lc_pad); // length npts
  dTyp *d_w            = (dTyp *)   (d_y          +   npts + lc_pad); // length npts
  dTyp *d_lsp          = (dTyp *)   (d_w          +   npts + lc_pad); // length ng/2 * (nbs + 1)
  filter_properties *d_fprops       = (filter_properties *)(d_lsp + (ng/2) * (nbs + 1) + lsp_pad);

  fprops->E1           = (dTyp *)   (d_fprops     +      1); // length npts
  fprops->E2           = (dTyp *)   (fprops->E1   +   npts + lc_pad); // length npts
  fprops->E3           = (dTyp *)   (fprops->E2   +   npts + lc_pad); // length FILTER_RADIUS  
  Complex *d_f_hat     = (Complex *)(fprops->E3   + (2 * (FILTER_RADIUS + 1))); // length ng * (nbs + 1) 
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
  cunfft_adjoint_raw_async(d_t,  d_y, d_f_hat,     npts,     ng, fprops, d_fprops, settings->stream);
  cunfft_adjoint_raw_async(d_t,  d_w, d_f_hat_win, npts, 2 * ng, fprops, d_fprops, settings->stream); 
  //checkCudaErrors(cudaStreamSynchronize(settings->stream));
  if (nbs > 0) {
    // set seed for random number generator
    seed = (unsigned int) clock();
    //printf("%lu\n", seed);

    // _TODO: I think it's a problem that the weights and the datapoints are shuffled independently.
    //       i.e. instead of shuffle([ (y, y_w), ... ]) we're doing shuffle([ y, y2, ...]) AND
    //       shuffle([ y_w, y_w2, ... ]). Obviously not a problem for constant weights, but things
    //       get hairy when that's not the case.
    // [FIXED 5/6/2016]

    cunfft_adjoint_raw_async_bootstrap(d_t, d_y, d_f_hat     + ng    , npts, ng    , 
				nbs, fprops, d_fprops, settings->stream, seed);
    cunfft_adjoint_raw_async_bootstrap(d_t, d_w, d_f_hat_win + 2 * ng, npts, 2 * ng, 
    				nbs, fprops, d_fprops, settings->stream, seed);
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

__host__ dTyp sgn(dTyp x) {
	if (x < 0) return -1.0;
	return 1.0;
}

// FROM: http://www.spraak.org/documentation/doxygen/src/lib/math/erfinv.c/view
dTyp spr_math_fast_erfinv(dTyp x)
/*
Calculate the inverse error function.
The fast version is only correct up to 6 digits.
*/
{dTyp tmp;
 int   neg;
 /**/
 if((neg=(x < 0.0)))
   x = -x;
 if(x <= 0.7)
  {tmp = x*x;
   x *= (((-0.140543331*tmp+0.914624893)*tmp-1.645349621)*tmp+0.886226899)/((((0.012229801*tmp-0.329097515)*tmp+1.442710462)*tmp-2.118377725)*tmp+1.0);
  }
 else
  {tmp = sqrt(-log(0.5*(1.0-x)));
   x = (((1.641345311*tmp+3.429567803)*tmp-1.624906493)*tmp-1.970840454)/((1.637067800*tmp+3.543889200)*tmp+1.0);
  }
 return(neg?-x:x);
}
//////////////////////////////////////////////////////////


/* approximation for inverse erf function (Wikipedia)
__host__ dTyp erfinverse(dTyp x) {
	dTyp inva = 1.0/0.147;
	dTyp b = 2.0 * inva / PI;
	dTyp c = log(1 - x * x);
	dTyp A = b + 0.5 * c;
	dTyp B = c * inva;
	return sgn(x) * sqrt(sqrt( A * A - B ) - A);
}*/

// getPnCutoff for bootstraps (need erfinv function that is not in math.h)
__host__ dTyp getPnCutoffBootstrap(dTyp proba, dTyp mu, dTyp sig) {
	return mu - sqrt(2) * sig * spr_math_fast_erfinv(2 * proba - 1);
}


// inverts the above formulae to provide the Pn value associated with a given FAP
__host__ dTyp getPnCutoff(dTyp proba, int npts, int nfreqs, dTyp over) {
	if (proba < EPSILON) {
		dTyp a = 0.5 * over * proba / nfreqs;
		dTyp b = 1 - pow(a, 2./(npts - 3));
		return 0.5 * b * ((dTyp)npts / ( npts - 1.0 ));
	} else {
		dTyp a = pow(1 - proba, 0.5 * over / nfreqs);
		dTyp b = pow(1 - a, 2./(npts - 3));
		return 0.5 * npts / (npts - 1) * (1 - b);
	}
}

