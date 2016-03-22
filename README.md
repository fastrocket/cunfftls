# CUNFFTLS v1.2: CUDA Lomb Scargle implementation

### (c) 2016, John Hoffman
### jah5@princeton.edu

##### Requires 
* [CUDA](https://developer.nvidia.com/cuda-toolkit) and compatible GPU(s); developed using the `CUDA-7.5` toolkit
* [CUNA](https://github.com/johnh2o2/cunfft_adjoint): CUDA implementation of the NFFT adjoint operation
* [Argtable2](http://argtable.sourceforge.net/): to read command line arguments
* [OpenMP](http://openmp.org)

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

* Added multi-thread, multi-gpu support with `--nthreads` command-line argument; automatically load-balances work among GPU's
* Currently fairly inefficient, will be optimizing next

#### Command line options

```
$ ./cunfftls --help
Usage: ./cunfftls  [-hvsd] [--version] [--in=<filename_in>] [--list-in=<list_in>] [--out=<filename_out>] [--list-out=<list_out>] [--over=<oversampling>] [--hifac=<hifac>] [--thresh=<hifac>] [--device=<device>] [--nthreads=<nthreads>] [--pow2] [--print-timing]

./cunfftlsf uses the NFFT adjoint operation to perform fast Lomb-Scargle calculations on GPU(s).

   -h, --help                     display usage/options
   --version                      display version
   --in=<filename_in>             input file
   --list-in=<list_in>            filename containing list of input files
   --out=<filename_out>           output file
   --list-out=<list_out>          filename to save peak LSP for each lightcurve
   --over=<oversampling>          oversample factor
   --hifac=<hifac>                max frequency = hifac * nyquist frequency
   --thresh=<hifac>               will save lsp if and only if the false alarm probability is below 'thresh'
   --device=<device>              device number (setting this forces this device to be the *only* device used)
   --nthreads=<nthreads>          number of openmp threads (tip: use a number >= number of GPU's)
   --pow2, --power-of-two         force nfreqs to be a power of 2
   --print-timing                 print calculation times
   -v, --verbose                  more output
   -s, --save-maxp                save max(LSP) for all lightcurves
   -d, --dont-save-lsp            do not save full LSP
 ```

#### Mode 1: single lightcurve
**Usage**:

```
./cunfftls --in=<filename_in> --out=<filename_out> [OPTS]

```

Lightcurves are expected to be in the form:

```
N
tobs_1 yobs_1
tobs_2 yobs_2
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

