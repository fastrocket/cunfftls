
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
  int lineno = 0, ncols, ncount;
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
  
  // allocate memory
  *tobs = (dTyp *)malloc(*npts * sizeof(dTyp));
  *yobs = (dTyp *)malloc(*npts * sizeof(dTyp));
  
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

void bootstrapLSP(const dTyp *tobs, const dTyp *yobs, const dTyp *errs, 
                  const int npts, Settings *settings, dTyp *mu, dTyp *sig) {
    // allocate
    dTyp *lsp_temp, *er;
    int mind;
    dTyp *maxvals  = (dTyp *)malloc(settings->nbootstraps * sizeof(dTyp));
    dTyp *t = (dTyp *) malloc(npts * sizeof(dTyp));
    dTyp *y = (dTyp *) malloc(npts * sizeof(dTyp));
    if (errs != NULL)
      er = (dTyp *) malloc(npts * sizeof(dTyp));
    else
      er = NULL;

    for(int i = 0; i < settings->nbootstraps; i++) {
      // sample with replacement
      randomSample(npts, tobs, yobs, errs, t, y, er);

      // do lomb scargle
      if (settings->lsp_flags & FLOATING_MEAN)
        lsp_temp = generalizedLombScargle(t, y, er, npts, settings);
      else
        lsp_temp = lombScargle(t, y, npts, settings);

      // find max and store
      mind = argmax(lsp_temp, settings->nfreqs);
      maxvals[i] = lsp_temp[mind];
    }

    // get statistics (mean and variance) of bootstrapped peaks
    meanAndVariance(settings->nbootstraps, maxvals, mu, sig);

    // convert variance to stddev
    *sig = sqrt(*sig);

    // clean up after ourselves
    free(t); free(y); free(er); free(maxvals);

}

// read lightcurve file and compute periodogram
void lombScargleFromLightcurveFile(Settings *settings) {
  dTyp *tobs, *yobs, *lspt, *frqs, *errs;
  int npts;
  
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

  // compute frequencies
  frqs = getFrequencies(tobs, npts, settings);
  
  // compute periodogram
  if (settings->lsp_flags & FLOATING_MEAN)
    lspt = generalizedLombScargle(tobs, yobs, errs, npts, settings);
  else
    lspt = lombScargle(tobs, yobs, npts, settings);

  // transfer memory out of the workspace
  dTyp *lsp = (dTyp *) malloc(settings->nfreqs * sizeof(dTyp));
  memcpy(lsp, lspt, settings->nfreqs * sizeof(dTyp));

  // perform bootstrapping if requested
  dTyp mu, sig;
  if (settings->nbootstraps > 0)  
    bootstrapLSP(tobs, yobs, errs, npts, settings, &mu, &sig);  
  

  // find peak
  int max_ind = argmax(lsp, settings->nfreqs);

  // write peak
  dTyp fap;
  if (settings->nbootstraps > 0)
    fap    = erfc((lsp[max_ind] - mu) / sig);
  else
    fap    = probability(lsp[max_ind], npts, settings->nfreqs, 
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
    for(int i = 0; i < settings->nfreqs; i++) 
      fprintf(settings->out, "%e %e\n", frqs[i], lsp[i]);
  }
  // save peak + false alarm probability
  if(settings->lsp_flags & SAVE_MAX_LSP) 
    fprintf(settings->outlist, "%s %.5e %.5e\n", settings->filename_in, 
                frqs[max_ind], fap);
  

  free(tobs); free(yobs); free(frqs); free(lsp);
  if (errs != NULL)
    free(errs);
  
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
  char lightcurves[nlightcurves][STRBUFFER];
        for(int i = 0; i < nlightcurves; i++){
    ncount = fscanf(settings->inlist, "%s", lightcurves[i]);
    if (ncount == 0) {
      eprint("cannot read lightcurve number %d\n", i + 1);
      exit(EXIT_FAILURE);
    }
  }
  
  // if we're saving peak values to another file, open that file now.
  if (settings->lsp_flags & SAVE_MAX_LSP) 
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

    int lcno = thread;
    while(lcno < nlightcurves){

      // set the lightcurve filename
      sprintf(thread_settings->filename_in, "%s", lightcurves[lcno]);
    
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
  struct arg_dbl *thresh    = arg_dbl0(NULL, "thresh",    "<fap_threshold>",         
          "will save lsp if and only if the false alarm probability"
          " is below 'thresh'");
  struct arg_int *mem       = arg_int0("m", "memory-per-thread", "<MB>",
          "workspace (pinned) memory allocated "
          "for each thread/stream");
  struct arg_int *dev       = arg_int0(NULL, "device",   "<device>", 
          "device number (setting this forces this device to be"
          " the *only* device used)");
  struct arg_int *nth       = arg_int0(NULL, "nthreads", "<nthreads>",
          "number of openmp threads "
          "(tip: use a number >= number of GPU's)");
  struct arg_int *btstrp    = arg_int0("b", "nbootstraps","<nbootstraps>",
          "number of bootstrapped samples to use for significance testing");  
  // flags
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
           hifac, thresh, dev, mem, nth, btstrp, pow2, flmean, timing, verb, 
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


  ////////////////////
  // parameters
  int MB = 1048576;
  settings->over0       = over->count   == 1 ? (dTyp) over->dval[0]   :   1;
  settings->hifac       = hifac->count  == 1 ? (dTyp) hifac->dval[0]  :   1;
  settings->device      = dev->count    == 1 ? (int)  dev->ival[0]    :  -1;
  settings->nthreads    = nth->count    == 1 ? (int)  nth->ival[0]    :   1;
  settings->fthresh     = thresh->count == 1 ? (dTyp) thresh->dval[0] : 0.0;
  settings->nbootstraps = btstrp->count == 1 ? (int) btstrp->ival[0]  :   0;
  // set workspace memory
  int total_memory   = mem->count    == 1 ? (int)  mem->ival[0]*MB : 512 * MB;
  int x = settings->over0 * settings->hifac;
  dTyp dtof = (4 + 6 * x) / (2 + 0.5 * x);
  settings->host_memory = (int) (total_memory / (1.0 + dtof));
  settings->device_memory = total_memory - settings->host_memory;

  if (settings->lsp_flags & VERBOSE) {
    fprintf(stderr, "%-20s = %f\n", "over", settings->over0);
    fprintf(stderr, "%-20s = %f\n", "hifac", settings->hifac);
    fprintf(stderr, "%-20s = %d\n", "device", settings->device);
    fprintf(stderr, "%-20s = %d\n", "nthreads", settings->nthreads);
    fprintf(stderr, "%-20s = %f\n", "thresh", settings->fthresh);
    fprintf(stderr, "%-20s = %d MB\n", "memory (host)", settings->host_memory/MB);
    fprintf(stderr, "%-20s = %d MB\n", "memory (device)", settings->device_memory/MB);
    fprintf(stderr, "%-20s = %d\n", "nbootstraps", settings->nbootstraps);
  }
    
  // RUN 
  init(settings, list_in);
  launch(settings, list_in);
  finish(settings, list_in);
  
  arg_freetable(argtable,sizeof(argtable)/sizeof(argtable[0]));

  return EXIT_SUCCESS;
}
