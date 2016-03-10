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


dTyp * getFrequencies(dTyp *x, dTyp over, int n, int ng) {
	dTyp range = x[n - 1] - x[0];
	dTyp df = 1. / (over * range);
	dTyp *freqs = (dTyp *)malloc(ng * sizeof(dTyp));
	for (int i = 0; i < ng; i++)
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
	fprintf(stderr, "%d observations...\n", *npts);

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
	LOG("readLC");
	readLC(in, &tobs, &yobs, &npts);
	LOG("calculate lomb scargle");
	lsp = lombScargle(tobs, yobs, npts, over, hifac, &nfreqs, NO_FLAGS);
	LOG("get frequencies");
	frqs = getFrequencies(tobs, over, npts, nfreqs);
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

int main(int argc, char *argv[]) {
	if (argc < 5) {
		fprintf(stderr, "usage:\n");
		fprintf(stderr, "       [ls from file   ]  (1) %s f <in>   <out>  <over> <hifac>\n", argv[0]);
		fprintf(stderr, "       [lomb scargle   ]  (2) %s l <n>    <over> <hifac>\n", argv[0]);
		fprintf(stderr, "       [lomb sc. timing]  (3) %s L <nmin> <nmax> <ntests>\n\n", argv[0]);
		fprintf(stderr, "n      : number of data points\n");
		fprintf(stderr, "nmin   : Smallest data size\n");
		fprintf(stderr, "nmax   : Largest data size\n");
		fprintf(stderr, "ntests : Number of runs\n");
		fprintf(stderr, "over   : oversampling factor\n");
		fprintf(stderr, "hifac  : high frequency factor\n");
		exit(EXIT_FAILURE);
	}

	if (argv[1][0] == 'l')
		testLombScargle(atoi(argv[2]), atof(argv[3]), atof(argv[4]));
	else if (argv[1][0] == 'L')
		timeLombScargle(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
	else if (argv[1][0] == 'f') {
		LOG("opening lightcurve file");
		FILE *lightcurve_file = fopen(argv[2], "r");
		FILE *lsp_file        = fopen(argv[3], "w");
		LOG("calculating lomb scargle");
		lombScargleFromLC(lightcurve_file, atof(argv[4]), atof(argv[5]), lsp_file);
		LOG("done.");
		fclose(lsp_file);
		fclose(lightcurve_file);
	}
	else {
		fprintf(stderr, "What does %c mean? Should be either 'f', 'l', or 'L'.\n", argv[1][0]);
		exit(EXIT_FAILURE);
	}
	return EXIT_SUCCESS;
}

