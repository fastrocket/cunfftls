/*   cunfftls_utils.h
 *   ================ 
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
#ifndef CUNFFTLS_UTILS_H
#define CUNFFTLS_UTILS_H

#include "cunfftls_typedefs.h"

// sorting
typedef struct {
    int index;
    dTyp value;
} element;

int compare_elements_reverse(const void *a, const void *b);

__host__ void argsort(dTyp *arr, int *inds, int nmembers);

// peak finding
__global__ void findPeaksGPU(const dTyp *x, int *p, const int n, const dTyp thresh);
__host__ void findPeaksCPU(const dTyp *x, int *p, int *npeaks, const int n, const dTyp thresh);

// sets CUDA device
__host__ void set_device(int device);

// ensures nfreqs is a power of 2, corrects oversampling accordingly
__host__ void getNfreqsAndCorrOversampling(int npts, Settings *settings);

// returns the next largest integer of the form 2^a where a \in (natural numbers)
__host__ dTyp nextPowerOfTwo(dTyp v);

// computes mean and variance of array y (of size n)
__host__ void meanAndVariance(const int n, const dTyp *y, dTyp *mean , dTyp *variance);

__host__ void weightedMeanAndVariance(const int n, const dTyp *y, 
								const dTyp *w, dTyp *mean, dTyp *variance);

// returns sign of a * abs(b)
__device__ dTyp sign(dTyp a, dTyp b);

// a * a
__device__ dTyp square(dTyp a);

// converts clock_t value into seconds
__host__ dTyp seconds(clock_t dt);

// generates random sample of (tobs, yobs, erobs) with replacement, 
// stores results in (t, y, er)
__host__ void randomSample(const int npts, const dTyp *tobs, const dTyp *yobs, 
					const dTyp *erobs, dTyp *t, dTyp *y, dTyp *er);

// find the index of maximum element
int inline argmax(const dTyp *x, const int n){
  int m = 0;
  for(int i = 0; i < n; i++) {
    if (x[i] > x[m]) {
      m = i;
    }
  }
  return m;
}

// find the maximum value of an array
dTyp inline maxval(const dTyp *x, const int n) {
  dTyp m = x[0];
  for(int i = 0; i < n; i++) {
    if (x[i] > m) {
      m = x[i];
    }
  }
  return m;
}

#endif
