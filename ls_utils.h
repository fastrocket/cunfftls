/*   ls_utils.h
 *   ========== 
 *   
 *   Misc. functions useful for other parts of the program 
 * 
 *   (c) John Hoffman 2016
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
 */

#ifndef LS_UTILS_H
#define LS_UTILS_H

#include "cuna_typedefs.h"
//#ifdef __CPLUSPLUS
//extern "C" {
//#endif
// returns the next largest integer of the form 2^a where a \in (natural numbers)
__host__ unsigned int nextPowerOfTwo(unsigned int v);

// computes mean and variance of array y (of size n)
__host__ void meanAndVariance(int n, const dTyp *y, dTyp *mean , dTyp *variance);

// returns sign of a * abs(b)
__device__ dTyp sign(dTyp a, dTyp b);

// a * a
__device__ dTyp square(dTyp a);

// converts clock_t value into seconds
__host__ dTyp seconds(clock_t dt);

// generates unequal timing array
__host__ dTyp * generateRandomTimes(int N);

// generates a periodic signal
__host__ dTyp * generateSignal(dTyp *x, dTyp f, dTyp phi, int N);

// checks if any nans are in the fft
__host__ int countNans(Complex *fft, int N);

// Rescale X to [0, 2pi)
__host__   void scale_x(dTyp *x, int size);

//OUTPUT UTILS
// GPU arrays
__host__   void printComplex_d( Complex *a, int N, FILE *out);
__host__   void printReal_d(    dTyp    *a, int N, FILE *out);
// CPU arrays
__host__   void printComplex(   Complex *a, int N, FILE *out);
__host__   void printReal(      dTyp    *a, int N, FILE *out);


// CUDA doesn't have a native atomic function if the variables are
// double precision, so we add an override here if we're doing double prec.
#ifdef DOUBLE_PRECISION
__device__ double atomicAdd(double* address, double val);
#endif
//#ifdef __cplusplus
//}
//#endif
#endif
