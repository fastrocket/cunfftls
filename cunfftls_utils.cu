/*   cunfftls_utils.cu
 *   =================
 *   
 *   Misc. functions useful for other parts of the program 
 * 
 *   (c) 2016, John Hoffman
 *   code borrowed extensively from B. Leroy's nfftls
 * 
 *   This file is part of cunfftls
 *
 *   cunfftls is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   cunfftls is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with cunfftls.  If not, see <http://www.gnu.org/licenses/>.
 *  
 *   Copyright (C) 2016, John Hoffman, 2012 B. Leroy [nfftls]
 */

#include <stdlib.h>
#include "cuna_utils.h"
#include "cuna_filter.h"
#include "cunfftls_utils.h"
#include "cunfftls_typedefs.h"


__host__ void getNfreqsAndCorrOversampling(int npts, Settings *settings){
   dTyp nfreqsr = 0.5 * npts * settings->over0 * settings->hifac;

   // correct the "oversampling" parameter accordingly
   settings->nfreqs = (int) nextPowerOfTwo(floor(nfreqsr));
   settings->over   = settings->over0 * settings->nfreqs / nfreqsr;

}

__global__ void
convertToComplex(dTyp *a, Complex *c, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		c[i].x = a[i];
		c[i].y = 0;
    }
}

__host__ dTyp 
nextPowerOfTwo(dTyp v) {
	return round(pow(2, round(log10(v) / log10(2.0)) + 1));
}

__host__ void
set_device(int device) {
	checkCudaErrors(cudaSetDevice(device));
}

__host__ void
meanAndVariance(const int n, const dTyp *y, dTyp *mean , dTyp *variance) {
  *mean = 0;
  dTyp M2 = 0, delta;
  
  int nn = 1;
  for(int i = 0; i < n; i++, nn++) {
    delta = y[i] - *mean;
    *mean += delta / nn;
    M2 += delta * (y[i] - *mean);
  }
  *variance = M2/(n - 1);
}

__host__ void
weightedMeanAndVariance(const int n, const dTyp *y, const dTyp *w, 
                        dTyp *mean, dTyp *variance) {
  *mean = 0;
  *variance = 0;

  for(int i = 0; i < n; i++) 
    *mean += w[i] * y[i];
  
  for(int i = 0; i < n; i++) 
    *variance += w[i] * (y[i] - *mean) * (y[i] - *mean);
  
}

__device__ dTyp
sign(dTyp a, dTyp b) {
  	return ((b >= 0) ? 1 : -1) * absoluteValueReal(a);
}

__device__ dTyp
square(dTyp a) { 
	return a * a; 
}


// converts clock_t value into seconds
__host__ dTyp 
seconds(clock_t dt) {
	return ((dTyp) dt) / ((dTyp)CLOCKS_PER_SEC);
}

__host__ void
randomSample(const int npts, const dTyp *tobs, const dTyp *yobs, 
              const dTyp *erobs, dTyp *t, dTyp *y, dTyp *er) {
  int u;
  if (erobs == NULL) er = NULL;
  
  for (int i = 0; i < npts; i++) {
    u = rand() % npts;
    t[i] = tobs[u];
    y[i] = yobs[u];
    if (erobs != NULL)
      er[i] = erobs[u];
  }
}
