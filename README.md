# CUNFFTLS: CUDA Lomb Scargle implementation

### (c) 2016, John Hoffman
### jah5@princeton.edu

##### Requires 
* [CUDA](https://developer.nvidia.com/cuda-toolkit) and compatible GPU(s)
	* Developed using the `CUDA-7.5` toolkit.
* [CUNA](https://github.com/johnh2o2/cunfft_adjoint)
* [Argtable2](http://argtable.sourceforge.net/)

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

#### Command line options

```
./cunfftls 

 --in=<filename_in>        input file
 --list-in=<list_in>       filename containing list of input files
 --out=<filename_out>      output file
 --list-out=<list_out>     filename to save peak LSP for each lightcurve
 --over=<oversampling>     oversample factor
 --hifac=<hifac>           max frequency = hifac * nyquist frequency
 --device=<device>         device number
 --pow2, --power-of-two    Force nfreqs to be a power of 2
 --print-timing            Print calculation times
 -v, --verbose             more output
 -s, --save-maxp           Save max(LSP) for all lightcurves
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

