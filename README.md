# CUNFFTLS: CUDA Lomb Scargle implementation

### (c) 2016, John Hoffman
### jah5@princeton.edu

Requires [CUNA](https://github.com/johnh2o2/cunfft_adjoint).

#### Mode 1: single lightcurve
**Usage**:

```
./cunfftls f input_file output_file over hifac
```

Where `input_file` is the path to the lightcurve, `output_file`
is the location to save the periodogram, `over` is an oversampling
parameter (multiplicative factor in the frequency resolution),
and `hifac` is a multiplicative factor in the maximum frequency cutoff.

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
./cunfftls F input_file over hifac
```

Where `input_file` is the path to the file containing a list of
lightcurve locations, `over` is the oversampling factor, and
`hifac` is a high frequency cutoff factor.

The file containing the list of lightcurves is expected to be
in the form:

```
N
path/to/lc1.dat
path/to/lc2.dat
...
```

where `N` is the number of lightcurves in the list.


