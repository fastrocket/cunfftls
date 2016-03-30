# CUNFFTLS v1.6: CUDA Lomb Scargle implementation

### (c) 2016, John Hoffman
### jah5@princeton.edu

##### Requires 
* [CUDA](https://developer.nvidia.com/cuda-toolkit) and compatible GPU(s); developed using the `CUDA-7.5` toolkit
* [OpenMP](http://openmp.org)
* [Argtable2](http://argtable.sourceforge.net/): to read command line arguments
* [cuna](https://github.com/johnh2o2/cunfft_adjoint): CUDA implementation of the NFFT adjoint operation

Uses the Non-equispaced Fast Fourier Transform (adjoint) to 
efficiently compute the Lomb-Scargle periodogram; this implements
the algorithm discussed in

>  B. Leroy,
>
>  "Fast calculation of the Lomb-Scargle periodogram using
>   nonequispaced fast Fourier transform"
>
>  Astron. Astrophys. 545, A50 (2012)
>  DOI: [http://dx.doi.org/10.1051/0004-6361/201219076](http://dx.doi.org/10.1051/0004-6361/201219076)

and borrows extensively from the associated codebase.

#### Recent changes
* **March 30, 2016**
   * Added a few sanity checks
   * Now, when running multiple threads and saving the peaks, each thread will write to their own unique file
* **March 29, 2016** (1.5)
   * Added ability to print out multiple peaks (`--npeaks` and `--peak-thresh`)
* **March 28, 2016** (1.4)
   * Fixed a memory alignment bug that arose with certain values for the number of observations
   * Did more testing, this time with randomly generated lightcurves containing Nobs values between 10 and 10,000; no bugs found
   * Memory management at this point is a headache; will try to tackle this in future releases.
* **March 26, 2016** (1.3)
   * fixed the bootstrapping method -- now bootstrapping is performed more efficiently, but there are two caveats currently:
	     1. None of this has been tested in a rigorous and complete way, so use at your own risk.
	     2. The random number generation is a little dubious, but very fast. Take a look at the **cuna** source code for more information.
   * The bootstrap calculations are also run on a single kernel, so memory limits the number of bootstraps that can currently be performed. Future versions will allow for multiple kernel calls to remove this constraint. 
* **March 23, 2016**
   * option to use floating-mean periodogram (aka the Generalized LSP a la [Zechmeister & Kuerster 2008](http://www.aanda.org/articles/aa/abs/2009/11/aa11296-08/aa11296-08.html))
   * Bootstrapped significance tests **NOTE** -- this feature is buggy right now, avoid until later updates
* Added multi-thread, multi-gpu support with `--nthreads` command-line argument; automatically load-balances work among GPU's
* Currently fairly inefficient, will be optimizing next

#### Command line options

```
$ ./cunfftls --help
Usage: ./cunfftls  [options]

./cunfftls uses the NFFT adjoint operation to perform fast Lomb-Scargle calculations on GPU(s).

-h, --help                display usage/options
      --version             display version
      --in=<string>         input file
      --list-in=<string>    filename containing list of input files
      --out=<string>        output file
      --list-out=<string>   filename to save peak LSP for each lightcurve
      --over=<float>        oversample factor
      --hifac=<float>       max frequency = hifac * nyquist frequency
      --npeaks=<int>        number of secondary peaks to save
      --peak-thresh=<float> FAP threshold for peak to be considered significant
      --thresh=<float>      will save lsp if and only if the false alarm probab
                            ility is below 'thresh'
      --device=<int>        device number (setting this forces this device to b
                            e the *only* device used)
  -m, --memory-per-thread=<float, in MB>
                            workspace (pinned) memory allocated for each thread
                            /stream
      --nthreads=<int>      number of openmp threads (tip: use a number >= numb
                            er of GPU's)
  -b, --nbootstraps=<int>   number of bootstrapped samples to use for significa
                            nce testing
      --pow2, --power-of-two
                            force nfreqs to be a power of 2
  -G, --floating-mean       use a floating mean (slightly slower, but more stat
                            istically robust)
      --print-timing        print calculation times
  -v, --verbose             more output
  -s, --save-maxp           save max(LSP) for all lightcurves
  -d, --dont-save-lsp       do not save full LSP

 ```

#### Mode 1: single lightcurve
**Usage**:

```
./cunfftls --in=<filename_in> --out=<filename_out> [OPTS]

```

Lightcurves are expected to be in the form:

```
N
tobs_1 yobs_1 [err_1]
tobs_2 yobs_2 [err_2]
...
```

where `N` is the number of observations in the lightcurve.

#### Mode 2: list of lightcurves

**Usage**

```
./cunfftls --list-in=<list_in> [OPTS]
```

The file containing the list of lightcurves is expected to be
in the form:

```
N
path/to/lc1.dat
path/to/lc2.dat
...
```

where `N` is the number of lightcurves in the list.

#### Benchmarking

[coming soon...]

