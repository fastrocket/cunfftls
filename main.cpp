/*   main.cpp
 *   ========
 *
 *   Driver code -- computes + stores LSP for lightcurve (and for)
 *   list of lightcurves
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
#include <argtable2.h>
#include <string.h>

#include "cuna.h"
#include "cuna_utils.h"

#include "cunfftls_typedefs.h"
#include "cunfftls_utils.h"
#include "cunfftls_periodogram.h"

// calculate frequencies for the lomb-scargle periodogram
dTyp * getFrequencies(dTyp *x, int npts, Settings *settings) {
	dTyp range = x[npts - 1] - x[0];
	
	// force grid to be power of two if necessary
	// this will correct "over"
	settings->nfreqs = (int)floor(0.5 * settings->over * settings->hifac * npts);
	if (settings->lsp_flags & FORCE_POWER_OF_TWO)
		getNfreqsAndCorrOversampling(npts, settings);
	
	// df = nyquist frequency / over
	dTyp df = 1. / (settings->over * range);

	// allocate frequency array
	dTyp *freqs = (dTyp *)malloc(settings->nfreqs * sizeof(dTyp));

	// write frequencies
	for (int i = 0; i < settings->nfreqs; i++)
		freqs[i] = (i + 1) * df;

	return freqs;

}


// Read lightcurve from filestream
void readLC(FILE *in, dTyp **tobs, dTyp **yobs, int *npts) {

	LOG("reading lc file");

	// read number of observations (first line)
	int nfound = fscanf(in, "%d", npts);
	if (nfound < 1) {
		eprint("can't read the first line (containing nobs)\n");
	}
	
	LOG("read in number of observations");

	// allocate memory
	*tobs = (dTyp *)malloc(*npts * sizeof(dTyp));
	*yobs = (dTyp *)malloc(*npts * sizeof(dTyp));
	
	LOG("malloced tobs and yobs");

	// read line by line
	for(int i = 0; i < *npts; i++) {
		nfound = fscanf(in, FORMAT, (*tobs) + i, (*yobs) + i);	
		if (nfound < 2) {
			eprint("could not read line %d of "
				"lc file (only %d found), t[i] = %lf, y[i] = %lf\n",
				i+1, nfound, (*tobs)[i], (*yobs)[i]);
		}
	}
}

// find the index of maximum element
int argmax(dTyp *x, int n){
	dTyp max = x[0];
	int m = 0;
	for(int i = 0; i < n; i++)
		if (x[i] > max) {
			max = x[i];
			m = i;
		}

	return m;
}

// read lightcurve file and compute periodogram
void lombScargleFromLightcurveFile(Settings *settings) {
	dTyp *tobs, *yobs, *lsp, *frqs;
	int npts;
	
	clock_t start;

	// Read lightcurve
	START_TIMER;
	readLC(settings->in, &tobs, &yobs, &npts);
	STOP_TIMER("readLC", start);

	// compute frequencies
	START_TIMER;
	frqs = getFrequencies(tobs, npts, settings);
	STOP_TIMER("getFrequencies", start);
	
	// compute periodogram
	START_TIMER;
	lsp = lombScargle(tobs, yobs, npts, settings);
	STOP_TIMER("lombScargle", start);
	
	// find peak
	START_TIMER;
	int max_ind = argmax(lsp, settings->nfreqs);
	STOP_TIMER("argmax (find peak)", start);

	// write peak
	dTyp fap = probability(lsp[max_ind], npts, settings->nfreqs, settings->over);
	printf("%s: max freq: %.3e, false alarm probability: %.3e\n", 
							settings->filename_in, frqs[max_ind], fap);

	// save lomb scargle
	if(settings->lsp_flags & SAVE_FULL_LSP) {
		START_TIMER;
		for(int i = 0; i < settings->nfreqs; i++) 
			fprintf(settings->out, "%e %e\n", frqs[i], lsp[i]);
		STOP_TIMER("writing lsp", start);
	}
	// save peak + false alarm probability
	if(settings->lsp_flags & SAVE_MAX_LSP) 
		fprintf(settings->outlist, "%s %.5e %.5e\n", settings->filename_in, 
								frqs[max_ind], fap);
	

}


// reads a list of lightcurve filenames and computes lomb scargle
// for each of them
void multipleLombScargle(Settings *settings) {
	int nlightcurves;

	LOG("in multipleLombScargle");

	// read the number of files from the first line
	int ncount = fscanf(settings->inlist, "%d", &nlightcurves);
	if (ncount == 0) {
		eprint("provide the number of lightcurves before any filenames.\n");
	}
	
	LOG("read number of files from list");

	// if we're saving peak values to another file, open that file now.
	if (settings->lsp_flags & SAVE_MAX_LSP) 
		settings->outlist = fopen(settings->filename_outlist, "w");
	
	LOG("opened outlist");

	// iterate through lightcurves
	for(int fno = 0 ; fno < nlightcurves ; fno++){

		LOG("new lightcurve");

		// read filename
		ncount = fscanf(settings->inlist, "%s", settings->filename_in);
		if (ncount == 0) {
			eprint("cant read filename number %d.\n",fno + 1);
		}

		LOG("read in filename");

		if(settings->lsp_flags & VERBOSE)
			fprintf(stderr, "Doing lc %d of %d [%s]\n", fno + 1, 
									nlightcurves, settings->filename_in);

		// open lightcurve file
		settings->in = fopen(settings->filename_in, "r");
		LOG("opened lightcurve");

		// open lsp file if we're saving the LSP
		if (settings->lsp_flags & SAVE_FULL_LSP) {
			// write filename (TODO: allow user to customize)
			sprintf(settings->filename_out, "%s.lsp", settings->filename_in);

			// open lsp file
			settings->out = fopen(settings->filename_out, "w");
		}
		LOG("opened lsp file");

		// compute lomb scargle (which will also save lsp to file)
		lombScargleFromLightcurveFile(settings);

		LOG("computed lomb scargle");

		// close lightcurve file (and lsp file)
		fclose(settings->in);
		if (settings->lsp_flags & SAVE_FULL_LSP) 
			fclose(settings->out);
	}

	// close the file where peaks are stored
	if (settings->lsp_flags & SAVE_MAX_LSP) 
		fclose(settings->outlist);
}
	
// driver 	
void launch(Settings *settings, bool list_mode) {
	if (settings->device != 0)
		set_device(settings->device);
	if (list_mode)
		multipleLombScargle(settings);
	else {
		settings->in  = fopen(settings->filename_in, "r");
		LOG("opened lightcurve");
		//settings->out = stdout;
		settings->out = fopen(settings->filename_out, "w");
		LOG("opened output file");

		lombScargleFromLightcurveFile(settings);

		fclose(settings->out);
		fclose(settings->in);
	}
	
}

// help screen
void usage(void *argtable[], struct arg_end *end) {
	fprintf(stderr,"PROBLEMS:\n");
    arg_print_errors(stderr,end,"cunfftls");
    fprintf(stderr, "HELP:\n");
    arg_print_glossary(stderr, argtable, " %-25s %s\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]){
	Settings *settings = (Settings *) malloc(sizeof(Settings));
	////////////////////////////////////////////////////////////////////////
	// command line arguments

	// in
	struct arg_file *in       = arg_file0(NULL, "in",      "<filename_in>", 
					"input file");

	struct arg_file *inlist   = arg_file0(NULL, "list-in", "<list_in>", 
					"filename containing list of input files");
	// out
	struct arg_file *out      = arg_file0(NULL, "out",     "<filename_out>", 
						"output file");
	struct arg_file *outlist  = arg_file0(NULL, "list-out","<list_out>", 
					"filename to save peak LSP for each lightcurve");
	// params
	struct arg_dbl *over      = arg_dbl0(NULL, "over",     "<oversampling>", 
					"oversample factor");
	struct arg_dbl *hifac     = arg_dbl0(NULL, "hifac",    "<hifac>",         
					"max frequency = hifac * nyquist frequency");
	struct arg_int *dev       = arg_int0(NULL, "device",   "<device>", 
					"device number");

	// flags
	struct arg_lit *pow2      = arg_lit0(NULL, "pow2,power-of-two", 
					"Force nfreqs to be a power of 2");
	struct arg_lit *timing    = arg_lit0(NULL, "print-timing", 
					"Print calculation times");
	struct arg_lit *verb      = arg_lit0("v",  "verbose", 
					"more output");
	struct arg_lit *savemax   = arg_lit0("s",  "save-maxp",
						"Save max(LSP) for all lightcurves");
	struct arg_lit *dsavelsp  = arg_lit0("d",  "dont-save-lsp" , 
					"do not save full LSP");
	////////////////////////////////////////////////////////////////////////

	struct arg_end *end = arg_end(20);

	void *argtable[] = { in, inlist, out, outlist, over, hifac, dev, pow2, timing,
							verb, savemax, dsavelsp, end };
	// Parse the command line
	int n_error = arg_parse(argc, argv, argtable);
	bool list_in = false;
  
	if (n_error != 0) 
		usage(argtable, end);

	settings->lsp_flags = 0;
	settings->nfft_flags = 0;

	////////////////////////
	// input/output files
	if (inlist->count  == 1) {
		strcpy(settings->filename_inlist, inlist->filename[0]);
		settings->inlist = fopen(settings->filename_inlist, "r");
		list_in = true;
		fprintf(stderr, "%-20s = %s\n", "list-in", settings->filename_inlist);
	}
	if (in->count      == 1){
		strcpy(settings->filename_in, in->filename[0]);
		if (list_in){
			eprint("can't specify both <in> and <inlist>");
		}
		fprintf(stderr, "%-20s = %s\n", "in", settings->filename_in);
	}
	if( outlist->count == 1) {
		if (!list_in) {
			fprintf(stderr, "Must specify <inlist> if <outlist> is specified");
			usage(argtable, end);
		}
		strcpy(settings->filename_outlist, outlist->filename[0]);
		settings->outlist = fopen(settings->filename_outlist, "w");
		fprintf(stderr, "%-20s = %s\n", "list-out", settings->filename_outlist);
	}
	if (out->count     == 1){
		if (list_in) {
			fprintf(stderr, "can't specify both <inlist> and <out>");
			usage(argtable, end);
		}

		strcpy(settings->filename_out, out->filename[0]);
		settings->out = fopen(settings->filename_out, "w");
		fprintf(stderr, "%-20s = %s\n", "out", settings->filename_out);
	}

	// require at least one type of input file
	if (in->count + inlist->count != 1) {
		fprintf(stderr, "requires at least one of --in or --list-in\n");
		usage(argtable, end);
	}
	// if its a single file, require output location
	if (!list_in && out->count == 0){
		fprintf(stderr, "for a single lightcurve, you must specify both --in and --out\n");
		usage(argtable, end);
	}

	////////////////////
	// flags
	if (savemax->count    == 1)
	settings->lsp_flags  |= SAVE_MAX_LSP;
	if (dsavelsp->count   == 0)
	settings->lsp_flags  |= SAVE_FULL_LSP;
	if (pow2->count       == 1)
	settings->lsp_flags  |= FORCE_POWER_OF_TWO;
	if (timing->count     == 1)
	settings->lsp_flags  |= TIMING;
	if (verb->count       == 1)
	settings->lsp_flags  |= VERBOSE;

	if(settings->lsp_flags & TIMING)
	settings->nfft_flags |= PRINT_TIMING;

	////////////////////
	// parameters
	settings->over   = over->count  == 1 ? (dTyp) over->dval[0]  : 1;
	settings->hifac  = hifac->count == 1 ? (dTyp) hifac->dval[0] : 1;
	settings->device = dev->count   == 1 ? (int)  dev->ival[0]   : 0;
	fprintf(stderr, "%-20s = %f\n", "over", settings->over);
	fprintf(stderr, "%-20s = %f\n", "hifac", settings->hifac);
	fprintf(stderr, "%-20s = %d\n", "device", settings->device);
  
  	// RUN 
	launch(settings, list_in);

	return EXIT_SUCCESS;
}
