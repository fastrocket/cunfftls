/*   cunfftls_typedefs.h
 *   ===================
 *
 *   Global variable types and macros for the cunfftls operations
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

#ifndef CUNFFTLS_TYPEDEFS_H
#define CUNFFTLS_TYPEDEFS_H

// include typedefs from the CUNA library (adjoint NFFT)
#include "cuna_typedefs.h"

// max length of strings
#define STRBUFFER 200

// some precision-dependent definitions
#ifdef DOUBLE_PRECISION
   #define PRECISION      'd'
   #define THREE_COL_FMT  "%le %le %*le"
   #define TWO_COL_FMT    "%le %le"
#else
   #define PRECISION      'f'
   #define THREE_COL_FMT  "%e %e %*e"
   #define TWO_COL_FMT    "%e %e"
#endif

///////////////////
// timer utilities (TODO: make these safer and not macros)
#define START_TIMER \
    if(settings->lsp_flags & TIMING) start = clock()
#define STOP_TIMER(label, start)\
    if(settings->lsp_flags & TIMING)\
        fprintf(stderr, "[ %-50s L%-5d ] %-30s : %.4e(s)\n", __FILE__,\
         __LINE__, #label, seconds(clock() - start))
///////////////////

// small number for scaling the observed times
#define EPSILON 1e-5

// FORMAT for reading lightcurves (default is two columns)
#define FORMAT TWO_COL_FMT

// flags for cunfftls
typedef enum {
	SAVE_FULL_LSP = 0x01,
	SAVE_MAX_LSP = 0x02,
	FORCE_POWER_OF_TWO = 0x04,
	TIMING = 0x08,
	VERBOSE = 0x10,
	SAVE_IF_SIGNIFICANT = 0x20
} LSP_FLAGS;

// settings 
typedef struct {
	FILE *inlist, *in, *out, *outlist;
	char filename_inlist [STRBUFFER];
	char filename_in     [STRBUFFER];
	char filename_out    [STRBUFFER];
	char filename_outlist[STRBUFFER];
	int device, nfreqs, nthreads;
	dTyp over0, over, hifac, df, fthresh;
	unsigned int nfft_flags;
	unsigned int lsp_flags;
} Settings;
#endif
