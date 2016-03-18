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
#include <omp.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

#include "cuna.h"
#include "cuna_utils.h"

#include "cunfftls_typedefs.h"
#include "cunfftls_utils.h"
#include "cunfftls_periodogram.h"

// calculate frequencies for the lomb-scargle periodogram
dTyp * getFrequencies(dTyp *x, int npts, Settings *settings) {
	dTyp range = x[npts - 1] - x[0];
	
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
		exit(EXIT_FAILURE);
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
			exit(EXIT_FAILURE);
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
	
	// force grid to be power of two if necessary
        // this will correct "over" (and "nfreqs")
        if (settings->lsp_flags & FORCE_POWER_OF_TWO)
                getNfreqsAndCorrOversampling(npts, settings);
	else {
		settings->over   = settings->over0;
        	settings->nfreqs = (int)floor(0.5 * settings->over 
					* settings->hifac * npts);
	}

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
	dTyp fap = probability(lsp[max_ind], npts, settings->nfreqs, 
														settings->over);
	if (settings->lsp_flags & VERBOSE ) 
		fprintf(stderr, "%s: max freq: %.3e, false alarm probability: %.3e\n", 
							settings->filename_in, frqs[max_ind], fap);

	// save lomb scargle (1) if all lsp's are to be saved
	if(settings->lsp_flags & SAVE_FULL_LSP 
			// or (2) if only significant lsp's are to be saved and this
			//        lsp is significant
			|| (settings->lsp_flags & SAVE_IF_SIGNIFICANT 
				&& fap < settings->fthresh) ) {
		START_TIMER;
		for(int i = 0; i < settings->nfreqs; i++) 
			fprintf(settings->out, "%e %e\n", frqs[i], lsp[i]);
		STOP_TIMER("writing lsp", start);
	}
	// save peak + false alarm probability
	if(settings->lsp_flags & SAVE_MAX_LSP) 
		fprintf(settings->outlist, "%s %.5e %.5e\n", settings->filename_in, 
								frqs[max_ind], fap);
	

	free(tobs);
	free(yobs);
	free(lsp);
	free(frqs);
}


// reads a list of lightcurve filenames and computes lomb scargle for each
void multipleLombScargle(Settings *settings) {
	int nlightcurves;

        // get number of GPUs
	int ngpus;
	checkCudaErrors(cudaGetDeviceCount(&ngpus));

	// read the number of files from the first line
	int ncount = fscanf(settings->inlist, "%d", &nlightcurves);
	if (ncount == 0) {
		eprint("provide the number of lightcurves before any filenames.\n");
		exit(EXIT_FAILURE);
	}
	
	// if we're saving peak values to another file, open that file now.
	if (settings->lsp_flags & SAVE_MAX_LSP) 
		settings->outlist = fopen(settings->filename_outlist, "w");
	
	// iterate through lightcurves
	int thread, global_lcno = 0, lcno;
	omp_set_num_threads(settings->nthreads);
	
	// OpenMP + CUDA for master/slave parallelism
	#pragma omp parallel default(shared) private(thread, lcno, ncount) 
	{
		clock_t start;
		bool tstop = false;
		// set GPU for thread
		thread = omp_get_thread_num();

		// make a copy of the settings
		Settings *thread_settings = (Settings *)malloc(sizeof(Settings));
		memcpy(thread_settings, settings, sizeof(Settings));
		
		// if the user did not set the device number, set it ourselves
		if (thread_settings->device == -1)
			thread_settings->device = thread % ngpus;
		checkCudaErrors(cudaSetDevice(thread_settings->device));
		
		while(true){
			// read filename & increment lc count
			#pragma omp critical
			{
				lcno = global_lcno;
				if (lcno >= nlightcurves) 
					tstop = true;

				if (!tstop) {
					ncount = fscanf(settings->inlist, "%s", 
									thread_settings->filename_in);
					if (ncount == 0) {
						eprint("thread %d, gpu %d: "
							"cant read filename number %d.\n",
							thread, thread_settings->device, lcno + 1);
						exit(EXIT_FAILURE);
					}	
					global_lcno++;
				}
			}
			// break criterion
			if (tstop) break;
		
			if (thread_settings->lsp_flags & VERBOSE)
				fprintf(stderr, "thread %d, gpu %d: Doing lc %d of %d [%s]\n", 
						thread, thread_settings->device, lcno + 1, 
						nlightcurves, thread_settings->filename_in);
			
			// open lightcurve file
			thread_settings->in = fopen(thread_settings->filename_in, "r");

			// open lsp file if we're saving the LSP
			if (thread_settings->lsp_flags & SAVE_FULL_LSP) {
				
				// write filename (TODO: allow user to customize)
				sprintf(thread_settings->filename_out, "%s.lsp", 
										thread_settings->filename_in);
	
				// open lsp file
				thread_settings->out = 
							fopen(thread_settings->filename_out, "w");
			}
			
			// compute lomb scargle (which will also save lsp to file)
			lombScargleFromLightcurveFile(thread_settings);
	
			// close lightcurve file (and lsp file)
			fclose(thread_settings->in);
			if (thread_settings->lsp_flags & SAVE_FULL_LSP) 
				fclose(thread_settings->out);
		} 
		if (thread_settings->lsp_flags & VERBOSE)
			fprintf(stderr, "thread %d (GPU %d) finished.\n", 
											thread, thread % ngpus);
		free(thread_settings);
	}

	// close the file where peaks are stored
	if (settings->lsp_flags & SAVE_MAX_LSP) 
		fclose(settings->outlist);
}
	
// driver 	
void launch(Settings *settings, bool list_mode) {
	if (list_mode)
		multipleLombScargle(settings);
	else 
		lombScargleFromLightcurveFile(settings);

}

// free memory and close files
void finish(Settings *settings, bool list_mode) {

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaProfilerStop());

	if (!list_mode) {
		fclose(settings->out);
		fclose(settings->in);
	}	
	free(settings);
}

char progname[200];
// prints usage/help information
void help(void *argtable[], struct arg_end *end) {
	fprintf(stderr, "Usage: %s ", progname);
	arg_print_syntax(stdout, argtable, "\n\n");
	fprintf(stderr, "%s uses the NFFT adjoint operation to perform "
		"fast Lomb-Scargle calculations on GPU(s).\n\n", progname);
	arg_print_glossary(stdout, argtable, "   %-30s %s\n");
}

// initialize before calculating LSP
void init(Settings * settings, bool list_mode) {
	// set GPU
	if (settings->device != -1)
		checkCudaErrors(cudaSetDevice(settings->device));
	if (!list_mode) {
		settings->in  = fopen(settings->filename_in, "r");
		settings->out = fopen(settings->filename_out, "w");
	}
}

// prints problems with command line arguments
void problems(void *argtable[], struct arg_end *end) {
	arg_print_errors(stderr, end, "cunfftls");
	fprintf(stderr, "Try '%s --help' for more information\n", progname);
	exit(EXIT_FAILURE);
}

// prints version information
void version(){
	#ifdef DOUBLE_PRECISION
	printf("%s (double precision): version %s\n",progname, VERSION);
	#else
	printf("%s (single precision): version %s\n",progname, VERSION);
	#endif  
}

// (main)
int main(int argc, char *argv[]){
	Settings *settings = (Settings *) malloc(sizeof(Settings));
	sprintf(progname, "%s", argv[0]);
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
	struct arg_dbl *thresh    = arg_dbl0(NULL, "thresh",    "<hifac>",         
					"will save lsp if and only if the false alarm probability"
					" is below 'thresh'");
	struct arg_int *dev       = arg_int0(NULL, "device",   "<device>", 
					"device number (setting this forces this device to be"
					" the *only* device used)");
	struct arg_int *nth       = arg_int0(NULL, "nthreads", "<nthreads>",
					"number of openmp threads "
					"(tip: use a number >= number of GPU's)");
	// flags
	struct arg_lit *pow2      = arg_lit0(NULL, "pow2,power-of-two", 
					"force nfreqs to be a power of 2");
	struct arg_lit *timing    = arg_lit0(NULL, "print-timing", 
					"print calculation times");
	struct arg_lit *verb      = arg_lit0("v",  "verbose", 
					"more output");
	struct arg_lit *savemax   = arg_lit0("s",  "save-maxp",
					"save max(LSP) for all lightcurves");
	struct arg_lit *dsavelsp  = arg_lit0("d",  "dont-save-lsp" , 
					"do not save full LSP");
	struct arg_lit *hlp       = arg_lit0("h", "help", 
					"display usage/options");
	struct arg_lit *vers      = arg_lit0(NULL, "version", 
					"display version"); 
	////////////////////////////////////////////////////////////////////////

	struct arg_end *end = arg_end(20);

	void *argtable[] = { hlp, vers, in, inlist, out, outlist, over, 
			             hifac, thresh, dev, nth, pow2, timing, verb, 
			             savemax, dsavelsp, end };
	
	// check that argtable didn't raise any memory problems
	if (arg_nullcheck(argtable) != 0) {
        eprint("%s: insufficient memory\n",progname);
		fprintf(stderr, "Try '%s --help' for more information\n", progname);
        exit(EXIT_FAILURE);
    }

	// Parse the command line
	int n_error = arg_parse(argc, argv, argtable);
	bool list_in = false;
  
	if (n_error != 0) 
		problems(argtable, end);
	else if (hlp->count >= 1) {
		help(argtable, end);
		exit(EXIT_SUCCESS);
	}
	else if (vers->count >= 1) {
		version();
		exit(EXIT_SUCCESS);
	}

	settings->lsp_flags = 0;
	settings->nfft_flags = 0;

	////////////////////////
	// input/output files
	if (inlist->count  == 1) {
		strcpy(settings->filename_inlist, inlist->filename[0]);
		settings->inlist = fopen(settings->filename_inlist, "r");
		list_in = true;
	}
	if (in->count      == 1){
		strcpy(settings->filename_in, in->filename[0]);
		if (list_in){
			eprint("can't specify both <in> and <inlist>");
			fprintf(stderr, "Try '%s --help' for more information\n", progname);
			exit(EXIT_FAILURE);
		}
	}
	if( outlist->count == 1) {
		if (!list_in) {
			eprint("Must specify <inlist> if <outlist> is specified");
			fprintf(stderr, "Try '%s --help' for more information\n", progname);
			exit(EXIT_FAILURE);
		}
		strcpy(settings->filename_outlist, outlist->filename[0]);
		settings->outlist = fopen(settings->filename_outlist, "w");
	}
	if (out->count     == 1){
		if (list_in) {
			eprint("can't specify both <inlist> and <out>");
			fprintf(stderr, "Try '%s --help' for more information\n", progname);
			exit(EXIT_FAILURE);
		}

		strcpy(settings->filename_out, out->filename[0]);
		settings->out = fopen(settings->filename_out, "w");
	}

	// require at least one type of input file
	if (in->count + inlist->count != 1) {
		eprint("requires at least one of --in or --list-in\n");
		fprintf(stderr, "Try '%s --help' for more information\n", progname);
		exit(EXIT_FAILURE);
	}
	// if its a single file, require output location
	if (!list_in && out->count == 0){
		eprint("for a single lightcurve, you must specify both --in and --out\n");
		fprintf(stderr, "Try '%s --help' for more information\n", progname);
		exit(EXIT_FAILURE);
	}

	////////////////////
	// flags
	if (savemax->count    == 1)
		settings->lsp_flags  |= SAVE_MAX_LSP;
	if (dsavelsp->count   == 0 && thresh->count == 0)
		settings->lsp_flags  |= SAVE_FULL_LSP;
	if (pow2->count       == 1)
		settings->lsp_flags  |= FORCE_POWER_OF_TWO;
	if (timing->count     == 1)
		settings->lsp_flags  |= TIMING;
	if (verb->count       == 1)
		settings->lsp_flags  |= VERBOSE;
	if (thresh->count     == 1)
		settings->lsp_flags  |= SAVE_IF_SIGNIFICANT;
	
	if (settings->lsp_flags & TIMING)
		settings->nfft_flags |= PRINT_TIMING;



	////////////////////
	// parameters
	settings->over0    = over->count   == 1 ? (dTyp) over->dval[0]   :  1;
	settings->hifac    = hifac->count  == 1 ? (dTyp) hifac->dval[0]  :  1;
	settings->device   = dev->count    == 1 ? (int)  dev->ival[0]    : -1;
	settings->nthreads = nth->count    == 1 ? (int)  nth->ival[0]    :  1;
	settings->fthresh  = thresh->count == 1 ? (dTyp) thresh->dval[0] :0.0;

	if (settings->lsp_flags & VERBOSE) {
		fprintf(stderr, "%-20s = %f\n", "over", settings->over0);
		fprintf(stderr, "%-20s = %f\n", "hifac", settings->hifac);
		fprintf(stderr, "%-20s = %d\n", "device", settings->device);
		fprintf(stderr, "%-20s = %d\n", "nthreads", settings->nthreads);
		fprintf(stderr, "%-20s = %f\n", "thresh", settings->fthresh);
  	}
  	
	// RUN 
	init(settings, list_in);
	launch(settings, list_in);
	finish(settings, list_in);
	
	arg_freetable(argtable,sizeof(argtable)/sizeof(argtable[0]));

	return EXIT_SUCCESS;
}
