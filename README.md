# CUNFFTLS v1.3: CUDA Lomb Scargle implementation

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

* **March 23, 2016**
   * option to use floating-mean periodogram (aka the Generalized LSP a la [Zechmeister & Kuerster 2008](http://www.aanda.org/articles/aa/abs/2009/11/aa11296-08/aa11296-08.html))
   * Bootstrapped significance tests **NOTE** -- this feature is buggy right now, avoid until later updates
* Added multi-thread, multi-gpu support with `--nthreads` command-line argument; automatically load-balances work among GPU's
* Currently fairly inefficient, will be optimizing next

#### Command line options

```
$ ./cunfftls --help
Usage: ./cunfftlsf  [-hGvsd] [--version] [--in=<filename_in>] [--list-in=<list_in>] [--out=<filename_out>] [--list-out=<list_out>] [--over=<oversampling>] [--hifac=<hifac>] [--thresh=<hifac>] [--device=<device>] [-m <MB>] [--nthreads=<nthreads>] [--pow2] [--print-timing]

./cunfftlsf uses the NFFT adjoint operation to perform fast Lomb-Scargle calculations on GPU(s).

  -h, --help                display usage/options
      --version             display version
      --in=<filename_in>    input file
      --list-in=<list_in>   filename containing list of input files
      --out=<filename_out>  output file
      --list-out=<list_out> filename to save peak LSP for each lightcurve
      --over=<oversampling> oversample factor
      --hifac=<hifac>       max frequency = hifac * nyquist frequency
      --thresh=<hifac>      will save lsp if and only if the false alarm probab
                            ility is below 'thresh'
      --device=<device>     device number (setting this forces this device to b
                            e the *only* device used)
  -m, --memory-per-thread=<MB> 
                            workspace (pinned) memory allocated for each thread
                            /stream
      --nthreads=<nthreads> number of openmp threads (tip: use a number >= numb
                            er of GPU's)
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

