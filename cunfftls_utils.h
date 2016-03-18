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

// sets CUDA device
__host__ void set_device(int device);

// ensures nfreqs is a power of 2, corrects oversampling accordingly
__host__ void getNfreqsAndCorrOversampling(int npts, Settings *settings);

// returns the next largest integer of the form 2^a where a \in (natural numbers)
__host__ dTyp nextPowerOfTwo(dTyp v);

// computes mean and variance of array y (of size n)
__host__ void meanAndVariance(int n, const dTyp *y, dTyp *mean , dTyp *variance);

// returns sign of a * abs(b)
__device__ dTyp sign(dTyp a, dTyp b);

// a * a
__device__ dTyp square(dTyp a);

// converts clock_t value into seconds
__host__ dTyp seconds(clock_t dt);

#endif
