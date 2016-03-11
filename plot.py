import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

OVER = 5
HIFAC = 2
SIGMA = 4.0
PHI = 0.4
Params = dict( jitter = 0, N = 10000, freq = 100., hifac = HIFAC, over = OVER, phi = PHI, sigma = SIGMA, outfile='lsp.dat', binary='cunfftlsd', lcfile='lc.dat')

lspdt = np.dtype([ ('f', float), ('p', float)])

def noise(params):
	return params['sigma'] * np.random.normal(size=params['N'])

def jitter(params):
	return params['jitter'] * noise(params) / params['sigma']

def get_signal(params):
	x = np.sort(np.random.random(params['N']))
	
	omega = 2 * np.pi * params['freq']
	phi = params['phi']
	y  = np.cos(omega * np.multiply(x, 1 + jitter(params))  - phi) + noise(params)
	#y += 0.6 * np.cos(2 * omega * np.multiply(x, 1 + jitter(params)) - 2 * np.pi * np.random.random() )
	#y += 0.6 * np.cos(2 * np.pi * x * 1.31 * params['freq'] - 0.44 * params['phi'])
	return x, y

def save_signal(x, y, lcfile='lc.dat'):
	f = open(lcfile, 'w')
	f.write('%d\n'%(len(x)))
	for X, Y in zip(x, y):
		f.write('%e %e\n'%(X, Y))
	f.close()
	

def get_lsp(x, y, **kwargs):
	save_signal(x, y, lcfile=kwargs['lcfile'])
	os.system("./%s --in=%s --out=%s --over=%f --hifac=%f --print-timing"%(kwargs['binary'], kwargs['lcfile'], kwargs['outfile'], kwargs['over'], kwargs['hifac']))
	return np.loadtxt(kwargs['outfile'], dtype=lspdt)

def get_peak(lsp):
	i = np.argmax(lsp['p'])
	return lsp['f'][i]

def get_diff(params):
	x, y = get_signal(params)
	return get_peak(get_lsp(x, y, **params)) - params['freq'] 

def test(params, frequencies):
	diffs = []
	for f in frequencies:
		params['freq'] = f
		diffs.append(get_diff(params))
	return diffs

def phase_fold(x, freq):
	return [ X*freq - int(X * freq) for X in x ]

def test_single_double(params):
	x, y = get_signal(params)
	fraw, axr = plt.subplots()
	axr.scatter(phase_fold(x, params['freq']), y, alpha=0.1, marker='.')
	fraw.savefig('lcraw.png')

	params['binary'] = 'cunfftlsf'
	single_lsp = get_lsp(x, y, **params)
	params['binary'] = 'cunfftlsd'
	double_lsp = get_lsp(x, y, **params)
	f, ax = plt.subplots()
	ax.plot(single_lsp['f'], single_lsp['p'], label='single', color='r', ls=':', alpha=0.5)
	ax.plot(double_lsp['f'], double_lsp['p'], label='double', color='k', alpha=0.5)
	ax.legend(loc='best')
	#d = 0.1
	#ax.set_xlim(params['freq'] * ( 1 - d ), params['freq'] * (1 + d))
	ax.set_ylim(0, 1.5 * max(double_lsp['p']))
	ax.axvline(params['freq'], color='b', ls='--')

	f.savefig('compare.png')

	plt.show()


Nlcs = 10
def test_list_of_files(params):
	l = open("list.dat",'w')
	l.write("%d\n"%(Nlcs))
	for i in range(	Nlcs ):
		x, y = get_signal(params)
		fname = "lcs/lc%04d.lc"%(i)
		
		save_signal(x, y, fname)
		l.write("%s\n"%fname)
	l.close()	
	
	os.system("./%s --list-in=list.dat --over=%e --hifac=%e --list-out=outlist.dat --print-timing"%(params['binary'], params['over'], params['hifac']))

test_single_double(Params)
test_list_of_files(Params)
#freqs = np.logspace(0, 2, 10)
#diffs = test(Params, freqs)
#print diffs
#reldiffs = [ dF / F0 for F0,dF in zip(freqs, diffs) ]
#print reldiffs
#print np.mean(reldiffs), np.std(reldiffs), 
#print [ d * Params['over'] for d in diffs ]
 
#test_single_double(Params)

