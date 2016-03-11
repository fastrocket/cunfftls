/*   main.cpp
 *   ========
 *
 *   UNIT TESTING for the cunfftls operations
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

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <cuda.h>

#include "cuna.h"
#include "cuna_typedefs.h"
#include "cuna_utils.h"
#include "ls_utils.h"
#include "lombScargle.h"

#define FREQUENCY 120.0
#define PHASE_SHIFT 0.5
#define HIFAC 2 
#define OVERSAMPLING 10


void getNfreqsAndCorrOversampling(int npts, int over, int hifac, dTyp *new_over, int *nfreqs){

   unsigned int nfreqs0 = (unsigned int) floor(0.5 * npts * over * hifac);

   // correct the "oversampling" parameter accordingly
   *nfreqs = (int) nextPowerOfTwo(nfreqs0);
   *new_over = over * ((float) (*nfreqs)) / ((float) nfreqs0);

}

dTyp * getFrequencies(dTyp *x, int n, dTyp over, dTyp hifac) {
	dTyp range = x[n - 1] - x[0];
	dTyp over_new; 
	int nfreqs;
	getNfreqsAndCorrOversampling(n, over, hifac, &over_new, &nfreqs);
	dTyp df = 1. / (over_new * range);
	dTyp *freqs = (dTyp *)malloc(nfreqs * sizeof(dTyp));
	for (int i = 0; i < nfreqs; i++)
		freqs[i] = (i + 1) * df;
        // TODO: validate this -- I'm getting offsets in the frequency
	return freqs;

}

void testLombScargle(int n, dTyp over, dTyp hifac) {
	int ng;

	dTyp *t = generateRandomTimes(n);
	dTyp *y = generateSignal(t, 120., 0.5, n);
	dTyp *lsp = lombScargle(t, y, n, over, hifac, &ng, NO_FLAGS);
	dTyp *freqs = getFrequencies(t, over, n, ng);

	for (int i = 0; i < ng; i++)
		printf("%.4e %.4e\n", freqs[i], lsp[i]);

}

/*void readHATlc(char *filename, dTyp *x, dTyp *y) {


}*/
#ifndef DOUBLE_PRECISION
#define three_col_fmt "%e %e %*e"
#define two_col_fmt "%e %e"
#else
#define two_col_fmt "%le %le"
#define three_col_fmt "%le %le %*le"
#endif

#define fmt two_col_fmt
void readLC(FILE *in, dTyp **tobs, dTyp **yobs, int *npts) {
	LOG("reading number of points");
	int nptstemp;	
	int nfound = fscanf(in, "%d", &nptstemp);
	*npts = nptstemp;

	if (nfound < 1) {
		eprint("can't read the first line (containing nobs)\n");
	}
	LOG("malloc");
	*tobs = (dTyp *)malloc(*npts * sizeof(dTyp));
	*yobs = (dTyp *)malloc(*npts * sizeof(dTyp));
	
	for(int i = 0; i < *npts; i++) {
		//fprintf(stderr,"reading obs %d\n", i);
		nfound = fscanf(in, fmt, (*tobs) + i, (*yobs) + i);	
		if (nfound < 2) {
			eprint("could not read line %d of lc file (only %d found), t[i] = %lf, y[i] = %lf\n",i+1, nfound, (*tobs)[i], (*yobs)[i]);
		}
	}

}


void lombScargleFromLC(FILE *in, dTyp over, dTyp hifac, FILE *out) {
	dTyp *tobs, *yobs, *lsp, *frqs;
	int npts, nfreqs;
	//clock_t s
	LOG("readLC");
	readLC(in, &tobs, &yobs, &npts);
	LOG("calculate lomb scargle");
	
	lsp = lombScargle(tobs, yobs, npts, over, hifac, &nfreqs, NO_FLAGS);
	LOG("get frequencies");
	frqs = getFrequencies(tobs, npts, over, hifac);
	LOG("print results to stream");
	for(int i = 0; i < nfreqs; i++) 
		fprintf(out, "%e %e\n", frqs[i], lsp[i]);

}
	

void timeLombScargle(int nmin, int nmax, int ntests) {
	int dN  = (nmax - nmin) / (ntests);
	int ng;
	clock_t start, dt;
	int test_no = 1;
	for(int n = nmin; n < nmax && test_no <= ntests; n+=dN, test_no++) {
		dTyp *t = generateRandomTimes(n);
		dTyp *y = generateSignal(t, FREQUENCY, PHASE_SHIFT, n);
		start = clock();
		lombScargle(t, y, n, OVERSAMPLING, HIFAC, &ng, PRINT_TIMING);
		dt = clock() - start;
		printf("TEST %-10d; Ndata = %-10d; Nfreq = %-10d; dt = %.5e s\n", test_no, n, ng, seconds(dt));

	}
}

void performManyLSP(FILE *in, dTyp over, dTyp hifac) {
	int nlightcurves;
	char filename[200], outname[200];
	int ncount = fscanf(in, "%d", &nlightcurves);
	
	if (ncount == 0) {
		eprint("provide the number of lightcurves before any filenames.\n");
	}
	LOG("read number of lightcurves");
	for(int fno = 0 ; fno < nlightcurves ; fno++){
		ncount = fscanf(in, "%s", filename);
		if (ncount == 0) {
			eprint("cant read filename number %d.\n",fno + 1);
		}
#ifdef DEBUG
		fprintf(stderr, "Doing lc %d of %d (%s)\n", fno + 1, nlightcurves, filename);
#endif
		FILE *in = fopen(filename, "r");
		sprintf(outname, "%s.lsp", filename);
		FILE *out = fopen(outname, "w");
		lombScargleFromLC(in, over, hifac, out);
		fclose(in);
		fclose(out);
	}
}

int main(int argc, char *argv[]) {
	FILE *input, *output;
	if (argc != 5 && argc != 6) {
		fprintf(stderr, "usage:\n");
		fprintf(stderr, "       [ls from file   ]  (1) %s f <in>     <out>  <over>  <hifac>\n", argv[0]);
		fprintf(stderr, "       [list of files  ]  (2) %s F <inlist> <over> <hifac>\n\n", argv[0]);
		fprintf(stderr, "in      : lightcurve filename\n");
		fprintf(stderr, "inlist  : filename containing list of lightcurve filenames\n");
		fprintf(stderr, "          (number of filenames on first line followed by filenames)\n");
		fprintf(stderr, "out     : filename to store periodogram\n");
		fprintf(stderr, "outroot : the prefix to the location of the periodograms\n");
		fprintf(stderr, "over    : oversampling factor\n");
		fprintf(stderr, "hifac   : high frequency factor\n");
		exit(EXIT_FAILURE);
	}
	switch(argv[1][0]) {
		case 'f':
			LOG("LSP from file");
			input  = fopen(argv[2], "r");
			output = fopen(argv[3], "w");
			lombScargleFromLC(input, atof(argv[4]), atof(argv[5]), output);
			fclose(input);
			fclose(output);
			break;
		case 'F' :
			LOG("list of lsps");
			input = fopen(argv[2], "r");
			performManyLSP(input, atof(argv[3]), atof(argv[4]));
			fclose(input);
			break;
		default:
			eprint( "What does %c mean? Should be either 'f', 'F'.\n", argv[1][0]);
			
	}
	return EXIT_SUCCESS;
}

