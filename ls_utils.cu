/*   utils.cu
 *   ========  
 *   
 *   
 * 
 *   (c) John Hoffman 2016
 * 
 *   This file is part of cuNFFT_adjoint
 *
 *   cuNFFT_adjoint is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   cuNFFT_adjoint is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with cuNFFT_adjoint.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include "cuna_utils.h"
#include "cuna_filter.h"
#include "ls_utils.h"

#define rmax 1000000
#define Random ((dTyp) (rand() % rmax))/rmax

#ifdef DOUBLE_PRECISION
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__global__ void
convertToComplex(dTyp *a, Complex *c, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		c[i].x = a[i];
		c[i].y = 0;
    }
}

__host__ unsigned int 
nextPowerOfTwo(unsigned int v) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

__host__ void
meanAndVariance(int n, const dTyp *y, dTyp *mean , dTyp *variance) {
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

// generates unequal timing array
__host__ dTyp * 
generateRandomTimes(int N) {
	dTyp *x = (dTyp *) malloc( N * sizeof(dTyp));
	x[0] = 0.;
	for (int i = 1; i < N; i++)
		x[i] = x[i - 1] + Random;

	dTyp xmax = x[N - 1];
	for (int i = 0; i < N; i++)
		x[i] = (x[i] / xmax) - 0.5;

	return x;
}

// generates a periodic signal
__host__ dTyp * 
generateSignal(dTyp *x, dTyp f, dTyp phi, int N) {
	dTyp *signal = (dTyp *) malloc( N * sizeof(dTyp));

	for (int i = 0; i < N; i++)
		signal[i] = cos((x[i] + 0.5) * f * 2 * PI - phi) + Random;

	return signal;
}


// checks if any nans are in the fft
int countNans(Complex *fft, int N) {
	int nans = 0;
	for (int i = 0 ; i < N; i++)
		if (isnan(fft[i].x) || isnan(fft[i].y))
			nans++;
	return nans;
}
__host__
void
printComplex_d(Complex *a_d, int N, FILE* out){
	Complex * cpu = (Complex *)malloc( N * sizeof(Complex));
	checkCudaErrors(cudaMemcpy(cpu, a_d, N * sizeof(Complex), cudaMemcpyDeviceToHost ));

	for(int i = 0;i < N; i++)
		fprintf(out, "%-5d %-10.3e %-10.3e\n", i, cpu[i].x, cpu[i].y);
        free(cpu);
}

__host__
void
printReal_d(dTyp *a, int N, FILE *out){
	dTyp * copy = (dTyp *) malloc(N * sizeof(dTyp));
        checkCudaErrors(cudaMemcpy(copy, a, N * sizeof(dTyp), cudaMemcpyDeviceToHost));

	for(int i = 0;i < N; i++)
		fprintf(out, "%-5d %-10.3e\n", i, copy[i]);
        free(copy);
}
__host__
void
printComplex(Complex *a, int N, FILE *out){
	for(int i = 0;i < N; i++)
		fprintf(out, "%-5d %-10.3e %-10.3e\n",  i, a[i].x, a[i].y);
}

__host__
void
printReal(dTyp *a, int N, FILE *out){
	for(int i = 0;i < N; i++)
		fprintf(out, "%-5d %-10.3e\n",  i, a[i]);
}

