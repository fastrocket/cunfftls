
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
#include <argtable2.h>
#include <omp.h>
#include <time.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

#include "cuna.h"
#include "cuna_utils.h"

#include "cunfftls_typedefs.h"
#include "cunfftls_utils.h"
#include "cunfftls_periodogram.h"

// program name
char progname[STRBUFFER];

// calculate frequencies for the lomb-scargle periodogram
dTyp * getFrequencies(dTyp *x, int npts, Settings *settings) {
  dTyp range = x[npts - 1] - x[0];

  // make sure range isn't 0 (not foolproof, but useful)
  if (range < 0) {
    eprint("lightcurve is not sorted by date\n");
    exit(EXIT_FAILURE);
  }  

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
void readLC(FILE *in, dTyp **tobs, dTyp **yobs, dTyp **errs, int *npts) {
  
  char buff[STRBUFFER];
  int lineno = 0, ncols = 0, ncount;
  dTyp err;

  // make sure file pointer is not NULL
  if (in == NULL) {
    eprint("input file is NULL.\n");
    exit(EXIT_FAILURE);
  }

  // read number of observations (first line)
  if ( fgets(buff, sizeof(buff), in) == NULL ) 
    ncount = 0;
  else 
    ncount = sscanf(buff, "%d", npts);
  
  // exit if unsuccessful
  if (ncount < 1) {
    eprint("can't read the first line (containing nobs)\n");
    exit(EXIT_FAILURE);
  }
  
  *tobs = (dTyp *)malloc(*npts* sizeof(dTyp));
  *yobs = (dTyp *)malloc(*npts* sizeof(dTyp));

  while (fgets(buff, sizeof(buff), in) != NULL) {

    // read first line to get number of columns
    if (lineno == 0) {
      ncols = sscanf(buff, THREE_COL_FMT, *tobs, *yobs, &err);
      if (ncols == 3) {
        *errs      = (dTyp *)malloc(*npts * sizeof(dTyp));
	      (*errs)[0] = err;
      }
      else if (ncols == 2)
        *errs = NULL;
      else {
      	eprint("ncols = %d, which is not 2 or 3.\n", ncols);
      	exit(EXIT_FAILURE);
      }
      lineno ++;
      continue;
    }

    // read line
    else if (ncols == 2) 
      ncount = sscanf(buff, TWO_COL_FMT, *tobs + lineno, *yobs + lineno);
    else if (ncols == 3) 
      ncount = sscanf(buff, THREE_COL_FMT, *tobs + lineno, *yobs + lineno, 
				*errs + lineno);

    // handle errors
    if (ncount != ncols) {
      eprint("problem reading line %d; only %d of %d "
	     "values matched (t[i] = %e, y[i] = %e, errs[i] = %e\n)", 
	      lineno + 1, ncount, ncols, (*tobs)[lineno], (*yobs)[lineno],
	      ncols == 3 ? (*errs)[lineno] : 0);
      exit(EXIT_FAILURE);
    } 
    lineno ++ ;
  }
}

// analyze set of bootstraps & store mean & standard deviation of max(powers)
void analyzeBootstraps(const dTyp *bootstraps, const int nfreqs, const int nlsp, dTyp *mu, dTyp *sig) {
   
   // allocate memory for maximum powers
   dTyp *maxp = (dTyp *) malloc( nlsp * sizeof(dTyp));
  
   // get max(power) for each bootstrap;
   for (int i = 0; i < nlsp; i++) 
      maxp[i] = maxval(bootstraps + i * nfreqs, nfreqs);

   // get mean and variance of max(powers)
   meanAndVariance(nlsp, maxp, mu, sig);
   
   // convert variance to standard deviation
   *sig = sqrt(*sig);

   // free array of max(powers)
   free(maxp);
}

#define sqrt2 1.41421356237
#define probabilityBootstrap(x, mu, sig) \
	0.5 * ( 1 + erf(-(x - mu) / (sqrt2 * sig)))


void getPeaks(dTyp *lsp, int **peaks, int *npeaks, dTyp cutoff, Settings *settings) {

  // find secondary peaks
  *peaks = NULL;
  *npeaks = 0;
  if (settings->npeaks == 0) return;

  // peak_check[i] is 1 when lsp[i] is a peak > threshold and 0 otherwise
  int *peak_check = (int *)malloc(settings->nfreqs * sizeof(int));

  // find the peaks	
  findPeaksCPU(lsp, peak_check, npeaks, settings->nfreqs, cutoff);

  // if only one peak (global max), return
  if (*npeaks == 1) {
    *npeaks = 0;
    return;
  }
  
  // store the indices
  *peaks = (int *) malloc(*npeaks * sizeof(int));
  int p = 0;
  for (int i = 0; i < settings->nfreqs; i++) 
    if ( peak_check[i] == 1 ) { (*peaks)[p] = i; p++; }
	
  // sort the peaks by significance (most to least significant)
  argsort(lsp, *peaks, *npeaks);
  
  // skip global max
  (*npeaks)--;
  for (int i = 0; i < *npeaks; i++)
     (*peaks)[i] = (*peaks)[i+1];
  
  // set the number of peaks to min(npeaks, settings->npeaks)	
  if (*npeaks > settings->npeaks)
    *npeaks = settings->npeaks;

  // free peak_check memory
  free(peak_check);
}

// read lightcurve file and compute periodogram
void lombScargleFromLightcurveFile(Settings *settings) {

  dTyp *tobs, *yobs, *lspt, *frqs, *errs;
  int npts;

  if (settings->lsp_flags & VERBOSE)
	  fprintf(stderr, "reading lightcurve\n");

  // Read lightcurve
  readLC(settings->in, &tobs, &yobs, &errs, &npts);
  
  // force grid to be power of two if necessary
  // this will correct "over" (and "nfreqs")
  if (settings->lsp_flags & FORCE_POWER_OF_TWO)
    getNfreqsAndCorrOversampling(npts, settings);
  else {
    settings->over   = settings->over0;
    settings->nfreqs = (int)floor(0.5 * settings->over 
          * settings->hifac * npts);
  }

  if (settings->lsp_flags & VERBOSE)
	  fprintf(stderr, "nfreqs = %d...getting frequencies\n",settings->nfreqs);

  // compute frequencies
  frqs = getFrequencies(tobs, npts, settings);
  
  if (settings->lsp_flags & VERBOSE)
	  fprintf(stderr, "performing lomb scargle\n");

  // compute periodogram
  if (settings->lsp_flags & FLOATING_MEAN)
    lspt = generalizedLombScargle(tobs, yobs, errs, npts, settings);
  else
    lspt = lombScargle(tobs, yobs, npts, settings);

  // wait for all processes to finish
  checkCudaErrors(cudaStreamSynchronize(settings->stream));

  if (settings->lsp_flags & VERBOSE)
	  fprintf(stderr, "copying memory from workspace\n");

  // transfer memory out of the workspace
  dTyp *lsp = (dTyp *) malloc(settings->nfreqs * (settings->nbootstraps + 1) * sizeof(dTyp));
  memcpy(lsp, lspt, settings->nfreqs * (settings->nbootstraps + 1) * sizeof(dTyp));


  //perform bootstrapping if requested
  dTyp mu, sig;
  if (settings->nbootstraps > 0) {
  	if (settings->lsp_flags & VERBOSE)
	  	fprintf(stderr, "bootstrapping...\n");
	  dTyp *bootstraps = lsp + settings->nfreqs;
	  analyzeBootstraps(bootstraps, settings->nfreqs, settings->nbootstraps, &mu, &sig);
  }

  if (settings->lsp_flags & VERBOSE)
	  fprintf(stderr, "finding peaks\n");

  // find max
  int mmax = argmax(lsp, settings->nfreqs);
  dTyp pbest = lsp[mmax];
  dTyp fbest = frqs[mmax];

  if (settings->lsp_flags & VERBOSE)
  	fprintf(stderr, "mmax = %d, pbest = %e, fbest = %e\n", mmax, pbest, fbest);

  // find secondary peaks
  int *peaks = NULL;
  int npeaks = 0;
  if (settings->npeaks > 0) {

     // cutoff is the Pn value associated with the seletced significance criterion
     dTyp cutoff;
     if (settings->nbootstraps > 0) {
        if (settings->lsp_flags & USE_SNR)
          cutoff = sig * settings->peak_significance + mu; 	
        else
          cutoff = getPnCutoffBootstrap(settings->peak_significance, mu, sig);
     } else
        cutoff = getPnCutoff(settings->peak_significance, npts, 
					settings->nfreqs, settings->over);
       
     getPeaks(lsp, &peaks, &npeaks, cutoff, settings);
     if (settings->lsp_flags & VERBOSE)
	     fprintf(stderr, "found %d peaks (%.5e, %.5e, ...)\n", npeaks, 
		                  npeaks > 0 ? frqs[peaks[0]] : -1,
		                  npeaks > 1 ? frqs[peaks[1]] : -1);
  }

  // calculate fap values
  dTyp fap;
  if (settings->nbootstraps > 0) {
    if (settings->lsp_flags & USE_SNR)
      fap = (pbest - mu) / sig;
    else 
      fap = probabilityBootstrap(pbest, mu, sig);
  } else
    fap   = probability(pbest, npts, settings->nfreqs, 
                         settings->over);
  
  if (settings->lsp_flags & VERBOSE ) {
    if (settings->lsp_flags & USE_SNR) 
      fprintf(stderr, "%s: max freq: %.3e, SNR: %.3e\n", 
              settings->filename_in, fbest, fap);
    else 
      fprintf(stderr, "%s: max freq: %.3e, false alarm probability: %.3e\n", 
              settings->filename_in, fbest, fap);
  }
  // save lomb scargle (1) if all lsp's are to be saved
  if(settings->lsp_flags & SAVE_FULL_LSP 
      // or (2) if only significant lsp's are to be saved and this
      //        lsp is significant
      || (
          settings->lsp_flags & SAVE_IF_SIGNIFICANT 
          && (
               (fap < settings->fthresh && !(settings->lsp_flags & USE_SNR)) 
            || (fap > settings->fthresh && (settings->lsp_flags & USE_SNR))
             )
         )) {

     settings->out = fopen(settings->filename_out, "w");
     if (settings->lsp_flags & USE_SNR)
       fprintf(settings->out, "#best freq | SNR\n"
			    "#%.5e %.5e\n", fbest, fap);  
     else
       fprintf(settings->out, "#best freq | false alarm probability\n"
			    "#%.5e %.5e\n", fbest, fap); 
     for(int i = 0; i < settings->nfreqs; i++) 
        fprintf(settings->out, "%e %e\n", frqs[i], lsp[i]);
     fclose(settings->out);
  }

  
  // save peak(s) + false alarm probability(s)
  if(settings->lsp_flags & SAVE_MAX_LSP) {
    fprintf(settings->outlist, "%s %.5e %.5e", settings->filename_in, 
                fbest, fap);

    dTyp frequency, false_alarm;
    for (int i = 0; i < settings->npeaks; i++) {
      if (npeaks <= i) {
        frequency = -1;
        false_alarm = -1;
      } else {
        frequency = frqs[peaks[i]];
        if (settings->nbootstraps > 0) {
          if (settings->lsp_flags & USE_SNR)
            false_alarm = lsp[peaks[i]] * sig + mu;
          else
            false_alarm = probabilityBootstrap(lsp[peaks[i]], mu, sig);
	} else
          false_alarm = probability(lsp[peaks[i]], npts, settings->nfreqs, settings->over);
      }
      fprintf(settings->outlist, " %.5e %.5e", frequency, false_alarm);
    }
    fprintf(settings->outlist, "\n");
  }
  
  if (settings->lsp_flags & VERBOSE)
	  fprintf(stderr, "freeing memory...\n");
  
  // free memory
  free(tobs); free(yobs); free(frqs); free(lsp);
  if (errs != NULL)
    free(errs);
  if (peaks != NULL) 
    free(peaks);
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

  // read entire list of lightcurves
  char **lightcurves = (char **)malloc(nlightcurves * sizeof(char *));

  for(int i = 0; i < nlightcurves; i++){
    lightcurves[i] = (char *)malloc(STRBUFFER * sizeof(char)); 
    ncount = fscanf(settings->inlist, "%s", lightcurves[i]);
    if (ncount == 0) {
      eprint("cannot read lightcurve number %d\n", i + 1);
      exit(EXIT_FAILURE);
    }
  }

  // if we're saving peak values to another file, open that file now.
  if (settings->lsp_flags & SAVE_MAX_LSP && settings->nthreads == 1) 
    settings->outlist = fopen(settings->filename_outlist, "w");
  
  // set number of OpenMP threads
  omp_set_num_threads(settings->nthreads);
  
  
  // OpenMP + CUDA
  #pragma omp parallel default(shared)
  {
    // get thread id
    int thread = omp_get_thread_num();

    // make a copy of the settings
    Settings *thread_settings = (Settings *)malloc(sizeof(Settings));
    memcpy(thread_settings, settings, sizeof(Settings));
    
    // if the user did not set the device number, set it ourselves
    if (thread_settings->device == -1)
      thread_settings->device = thread % ngpus;
    checkCudaErrors(cudaSetDevice(thread_settings->device));
    
    // make new stream for this thread
    checkCudaErrors(cudaStreamCreate(&(thread_settings->stream)));

    // allocate pinned and device memory for thread's workspace
    void *h_workspace, *d_workspace;
    checkCudaErrors(cudaHostAlloc((void **) &h_workspace, 
        thread_settings->host_memory, cudaHostAllocDefault));
    checkCudaErrors(cudaMalloc((void **) &d_workspace, 
        thread_settings->device_memory));
    thread_settings->device_workspace = d_workspace;
    thread_settings->host_workspace = h_workspace;

    // open maxp-results file for this specific thread
    if (thread_settings->lsp_flags & SAVE_MAX_LSP && thread_settings->nthreads > 1) {
        sprintf(thread_settings->filename_outlist, "%s.%d", 
				thread_settings->filename_outlist, thread);
        thread_settings->outlist = fopen(thread_settings->filename_outlist, "w");
    }

    int lcno = thread;
    while(lcno < nlightcurves){

      // set the lightcurve filename
      sprintf(thread_settings->filename_in, "%s", lightcurves[lcno]);
    
      if (thread_settings->lsp_flags & VERBOSE)
        fprintf(stderr, "thread %d, gpu %d: Doing lc %d of %d [%s]\n", 
            thread, thread_settings->device, lcno + 1, 
            nlightcurves, thread_settings->filename_in);
      
      // open lightcurve file
      if (!file_exists(thread_settings->filename_in)) {
        eprint("cannot find file: '%s'\n", thread_settings->filename_in);
        exit(EXIT_FAILURE);
      }
      thread_settings->in = fopen(thread_settings->filename_in, "r");

      // write filename (TODO: allow user to customize)
      sprintf(thread_settings->filename_out, "%s.lsp", 
              thread_settings->filename_in);

      // open lsp file if we're saving the LSP
      if (thread_settings->lsp_flags & SAVE_FULL_LSP)
        thread_settings->out = 
            fopen(thread_settings->filename_out, "w");
      
      // compute lomb scargle (which will also save lsp to file)
      lombScargleFromLightcurveFile(thread_settings);
  
      // close lightcurve file (and lsp file)
      fclose(thread_settings->in);
      if (thread_settings->lsp_flags & SAVE_FULL_LSP) 
        fclose(thread_settings->out);

      // increment lightcurve number
      lcno += settings->nthreads;
    } 
    if (thread_settings->lsp_flags & VERBOSE)
      fprintf(stderr, "thread %d (GPU %d) finished.\n", 
                thread, thread % ngpus);

    // free memory
    checkCudaErrors(cudaStreamDestroy(thread_settings->stream));
    checkCudaErrors(cudaFreeHost(thread_settings->host_workspace));
    checkCudaErrors(cudaFree(thread_settings->device_workspace));
    if (thread_settings->lsp_flags & SAVE_MAX_LSP)
      fclose(thread_settings->outlist);
    free(thread_settings);
  }
  

  // close the file where peaks are stored
  if (settings->lsp_flags & SAVE_MAX_LSP && settings->nthreads == 1) 
    fclose(settings->outlist);
        
}
  
// initialize before calculating LSP
void init(Settings * settings, bool list_mode) {

  // initialize random number generator for bootstrapping
  if (settings->nbootstraps > 0) 
    srand (time(NULL));

  // set GPU
  if (settings->device != -1) {
    checkCudaErrors(cudaSetDevice(settings->device));
  }
  if (!list_mode) {
    
    // open lightcurve file and lsp file
    if (!file_exists(settings->filename_in)) {
      eprint("cannot find file: '%s'\n", settings->filename_in);
      exit(EXIT_FAILURE);
    }
    settings->in  = fopen(settings->filename_in, "r");

    if (settings->lsp_flags & SAVE_FULL_LSP)
      settings->out = fopen(settings->filename_out, "w");
    else
      settings->out = stdout;

    // setting device if specified
    if (settings->device != -1) {
      checkCudaErrors(cudaSetDevice(settings->device));
    }
    else {
      checkCudaErrors(cudaSetDevice(0));
    }

    // set stream
    checkCudaErrors(cudaStreamCreate(&(settings->stream)));
    
    // allocate host and device workspaces
    checkCudaErrors(cudaHostAlloc((void **) &(settings->host_workspace), 
      settings->host_memory,  cudaHostAllocDefault ));
    checkCudaErrors(cudaMalloc((void **) &(settings->device_workspace),
      settings->device_memory));

    // memset workspaces to 0
    checkCudaErrors(cudaMemsetAsync(settings->device_workspace, 0, 
    	settings->device_memory, settings->stream));
    memset(settings->host_workspace, 0, settings->host_memory);

  }
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

  if (!list_mode) {
    // close files
    fclose(settings->in);
    if (settings->lsp_flags & SAVE_FULL_LSP)
      fclose(settings->out);

    // destroy stream
    checkCudaErrors(cudaStreamDestroy(settings->stream));

    // clean workspace
    checkCudaErrors(cudaFreeHost(settings->host_workspace));
    checkCudaErrors(cudaFree(settings->device_workspace));
  } 
  free(settings); 
  checkCudaErrors(cudaDeviceReset());
  checkCudaErrors(cudaProfilerStop());
}

// prints usage/help information
void help(void *argtable[]) {
  fprintf(stderr, "Usage: %s ", progname);
  arg_print_syntax(stdout, argtable, "\n\n");
  fprintf(stderr, "%s uses the NFFT adjoint operation to perform "
    "fast Lomb-Scargle calculations on GPU(s).\n\n", progname);
  arg_print_glossary_gnu(stdout, argtable);
}

// prints problems with command line arguments
void problems(struct arg_end *end) {
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

void sanity_check(Settings *settings) {
	if (settings->nthreads <= 0) {
		eprint("nthreads must be >= 1\n");
		exit(EXIT_FAILURE);
	}
	if (settings->lsp_flags & SAVE_MAX_LSP 
		&& !file_exists(settings->filename_inlist)) {
		eprint("inlist [%s] does not exist or could not be opened.\n");
		exit(EXIT_FAILURE);
	}
	if (settings->npeaks < 0) {
		eprint("npeaks must be >= 0\n");
		exit(EXIT_FAILURE);
	}
	if (settings->peak_significance < 0 || settings->peak_significance > 1){
		eprint("--peak-thresh must be between 0 and 1\n");
		exit(EXIT_FAILURE);
	}
}

// (main)
int main(int argc, char *argv[]){
  Settings *settings = (Settings *) malloc(sizeof(Settings));
  sprintf(progname, "%s", argv[0]);
  ////////////////////////////////////////////////////////////////////////
  // command line arguments

  // in
  struct arg_file *in       = arg_file0(NULL, "in",      "<string>", 
          "input file");

  struct arg_file *inlist   = arg_file0(NULL, "list-in", "<string>", 
          "filename containing list of input files");
  // out
  struct arg_file *out      = arg_file0(NULL, "out",     "<string>", 
          "output file");
  struct arg_file *outlist  = arg_file0(NULL, "list-out","<string>", 
          "filename to save peak LSP for each lightcurve");
  // params
  struct arg_dbl *over      = arg_dbl0(NULL, "over",     "<float>", 
          "oversample factor");
  struct arg_dbl *hifac     = arg_dbl0(NULL, "hifac",    "<float>",         
          "max frequency = hifac * nyquist frequency");
  struct arg_dbl *thresh    = arg_dbl0(NULL, "thresh",    "<float>",         
          "will save lsp if and only if the false alarm probability"
          " is below 'thresh'");
  struct arg_dbl *peaksig   = arg_dbl0(NULL, "peak-thresh", "<float>",
          "FAP threshold for peak to be considered significant");
  
  struct arg_int *peaks     = arg_int0(NULL, "npeaks", "<int>",
          "number of secondary peaks to save");
  struct arg_int *mem       = arg_int0("m", "memory-per-thread", "<float, in MB>",
          "workspace (pinned) memory allocated "
          "for each thread/stream");
  struct arg_int *dev       = arg_int0(NULL, "device",   "<int>", 
          "device number (setting this forces this device to be"
          " the *only* device used)");
  struct arg_int *nth       = arg_int0(NULL, "nthreads", "<int>",
          "number of openmp threads "
          "(tip: use a number >= number of GPU's)");
  struct arg_int *btstrp    = arg_int0("b", "nbootstraps","<int>",
          "number of bootstrapped samples to use for significance testing");  
  // flags
  struct arg_lit *snr       = arg_lit0(NULL, "use-snr", 
          "instead of FAP, use (power - <power>)/sqrt(<power^2>)");
  struct arg_lit *pow2      = arg_lit0(NULL, "pow2,power-of-two", 
          "force nfreqs to be a power of 2");
  struct arg_lit *flmean    = arg_lit0("G", "floating-mean", 
          "use a floating mean (slightly slower, "
          "but more statistically robust)");
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
           hifac, peaks, peaksig, thresh, dev, mem, nth, btstrp, 
           snr, pow2, flmean, timing, verb, savemax, dsavelsp, end };
  
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
    problems(end);
  else if (hlp->count >= 1) {
    help(argtable);
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
    sprintf(settings->filename_inlist, "%s", inlist->filename[0]);
    settings->inlist = fopen(settings->filename_inlist, "r");
    list_in = true;
  }
  if (in->count      == 1){
    sprintf(settings->filename_in, "%s", in->filename[0]);
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
    sprintf(settings->filename_outlist, "%s", outlist->filename[0]);
    settings->outlist = fopen(settings->filename_outlist, "w");
  }
  if (out->count     == 1){
    if (list_in) {
      eprint("can't specify both <inlist> and <out>");
      fprintf(stderr, "Try '%s --help' for more information\n", progname);
      exit(EXIT_FAILURE);
    }

    sprintf(settings->filename_out,"%s", out->filename[0]);
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
  if (flmean->count     == 1)
    settings->lsp_flags  |= FLOATING_MEAN;
  if (snr->count        == 1)
    settings->lsp_flags  |= USE_SNR;

  ////////////////////
  // parameters
  int MB = 1048576;
  settings->over0       = over->count   == 1 ? (dTyp) over->dval[0]   :   1;
  settings->hifac       = hifac->count  == 1 ? (dTyp) hifac->dval[0]  :   1;
  settings->device      = dev->count    == 1 ? (int)  dev->ival[0]    :  -1;
  settings->nthreads    = nth->count    == 1 ? (size_t)  nth->ival[0]    :   1;
  settings->fthresh     = thresh->count == 1 ? (dTyp) thresh->dval[0] : 0.0;
  settings->nbootstraps = btstrp->count == 1 ? (size_t) btstrp->ival[0]  :   0;
  settings->npeaks      = peaks->count  == 1 ? (int)  peaks->ival[0] : 0;
  settings->peak_significance = peaksig->count == 1 ? (dTyp) peaksig->dval[0] : 0;

  // set workspace memory
  size_t total_memory   = mem->count    == 1 ? (size_t)  mem->ival[0] : 512;
  dTyp dtoh;
  if (settings->lsp_flags & FLOATING_MEAN) {
    dtoh  = 5. + 13 * settings->over0 * (settings->nbootstraps + 1);
    dtoh /= 3. +      settings->over0 * (settings->nbootstraps + 1);
  } else {
    dtoh  = 2. +       settings->over0 * (2.5 * (settings->nbootstraps + 1) + 4);
    dtoh /= 1. + 0.5 * settings->over0 * (settings->nbootstraps + 1);
  }
  settings->host_memory = (size_t) (((dTyp) total_memory) / (1.0 + dtoh));
  settings->device_memory = total_memory - settings->host_memory;
  
  settings->host_memory *= MB;
  settings->device_memory *= MB;
  
  if (settings->lsp_flags & VERBOSE) {
    fprintf(stderr, "%-20s = %f\n", "over", settings->over0);
    fprintf(stderr, "%-20s = %f\n", "hifac", settings->hifac);
    fprintf(stderr, "%-20s = %d\n", "device", settings->device);
    fprintf(stderr, "%-20s = %d\n", "nthreads", settings->nthreads);
    fprintf(stderr, "%-20s = %f\n", "thresh", settings->fthresh);
    fprintf(stderr, "%-20s = %ld MB\n", "memory (host)", settings->host_memory/MB);
    fprintf(stderr, "%-20s = %ld MB\n", "memory (device)", settings->device_memory/MB);
    fprintf(stderr, "%-20s = %d\n", "nbootstraps", settings->nbootstraps);
  }
    
  // RUN 
  sanity_check(settings);
  init(settings, list_in);
  launch(settings, list_in);
  finish(settings, list_in);
  
  arg_freetable(argtable,sizeof(argtable)/sizeof(argtable[0]));

  return EXIT_SUCCESS;
}

