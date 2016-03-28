import numpy as np
import matplotlib.pyplot as plt
import os, sys
from math import *
import cPickle as pickle 

OVER = 10
HIFAC = 2
SIGMA = 4.0
PHI = 0.4
Params = dict( jitter = 0, N = 3000, freq = 100., 
		hifac = HIFAC,       over = OVER, phi = PHI, 
		sigma = SIGMA,     outfile='lsp.dat', 
		binary='cunfftlsf', lcfile='lc.dat',
		nthreads = 4
	)

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
	command = "./%s --in=%s --out=%s --over=%f --hifac=%f --nthreads=%d "%(kwargs['binary'], 
			kwargs['lcfile'], kwargs['outfile'], kwargs['over'], kwargs['hifac'], kwargs['nthreads'])
	print command
	os.system(command)
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

def plot_lsp(lsp, title=None, fname='lsp.png', params=None):
	f, ax = plt.subplots()
	ax.plot(lsp['f'], lsp['p'], color='k', alpha=0.5)
	ax.axvline(params['freq'], color='b', ls='--')
	ax.set_xlabel('freq')
	ax.set_ylabel('ls power')
	if not title is None:
		f.suptitle(title)
	f.savefig(fname)

def make_fake_lcs(params, Nlcs):
	l = open("list.dat", 'w')
        l.write('%d\n'%Nlcs)
        for i in range(Nlcs):
		x, y = get_signal(params)
		fname = "lcs/lc%05d.lc"%(i)
		save_signal(x, y, fname)
		l.write('%s\n'%fname)
	l.close()

def test_list_of_files(params):
	command = "./%s --verbose --list-in=list.dat --over=%e --hifac=%e --list-out=outlist.dat --dont-save-lsp --save-maxp --nthreads=%d --nbootstraps=50"%(
				params['binary'], params['over'], params['hifac'], params['nthreads'])
	print command
	os.system(command)


def test_floating_mean(params):
	
	command = "./%s --in=lcs/lc00001.lc --floating-mean --over=%e --hifac=%e --out=fmean_lsp.dat --nbootstraps=100 --verbose --memory-per-thread=1024"%(
				params['binary'], params['over'], params['hifac'])
	print command
	os.system(command)

	command = "./%s --in=lcs/lc00001.lc --over=%e --hifac=%e --out=no_fmean_lsp.dat --nbootstraps=100 --verbose"%(
				params['binary'], params['over'], params['hifac'])
	print command
	os.system(command)
	f, ax = plt.subplots()
	lsp_old = np.loadtxt('no_fmean_lsp.dat', dtype=lspdt)
	lsp_new = np.loadtxt('fmean_lsp.dat', dtype=lspdt)
	ax.axvline(params['freq'], ls='--', color='r')
	ax.plot(lsp_old['f'], lsp_old['p'], color='k', alpha=0.5, label="No floating mean")
	ax.plot(lsp_new['f'], lsp_new['p'], color='r', alpha=0.7, label="floating mean")
	ax.legend(loc='best')
	ax.set_xlabel("freq (1/d)")
	ax.set_ylabel("lsp power")
	plt.show()
	f.savefig("floating_mean_test.png")
	
def make_lc_with_gaps(params):
	x, y = get_signal(params)
	inds = range(len(x)/4)
	inds.extend(range(len(x)/2, len(x)))
	mask = [ True if i in inds else False for i in range(len(x)) ]
	
	return [ X for i,X in enumerate(x) if mask[i] ], [ Y for i, Y in enumerate(y) if mask[i] ]

def test_gap(params):
	x, y = make_lc_with_gaps(params)
	f, ax = plt.subplots()
	#p = phase_fold(x, params['freq'])
	ax.scatter(x, y, alpha=0.5, color='k', marker=',')
	f.savefig("raw_gap.png")

	lsp = get_lsp(x, y, **params)
	plot_lsp(lsp, title='gap', fname='lsp_gap.png', params=params)

def load_maxp_file(fname):
	dt = np.dtype([ ('lcfname', 'S100'), ('freq', float), ('fap', float)])
	return np.loadtxt(fname, dtype=dt)

def get_frecovered_as_function_of(results, params, variable, nbins=25, thresh=1E-2):
	res_dict = { r['lcfname'] : r for r in results }
	variables = [ params[fname][variable] for fname in params ]
	phi = [ abs(res_dict[fname]['freq']/params[fname]['freq'] - 1) for fname in params ]

	v0 = min(variables)
	dv = (max(variables) - v0)/nbins
	v_bins = [ v0 + (i + 0.5) * dv for i in range(nbins)]
	r_bins = np.zeros(nbins)
	tot_bins = np.zeros(nbins)
	e_bins = np.zeros(nbins)
	for var, p in zip(variables, phi):
		b = int((var - v0)/dv)
		if b == nbins: b = nbins - 1
		if p < thresh: r_bins[b] += 1
		tot_bins[b] += 1

	for i in range(nbins):
		t = tot_bins[i]
		r = r_bins[i]
		if t > 0: 
			r_bins[i] = r/t
			if r == 0: e_bins[i] = 1./t
			else:
				e_bins[i] = r_bins[i] * sqrt(pow(t, -1) + pow(r, -1))

	return v_bins, r_bins, e_bins
def inject_and_recover(params, nlc, do_lcs=False, nmin=10, nmax=10000, freqmin=0.01, freqmax=2000, sigmin=0.1, sigmax=10.0):
	all_pars = {}
	if do_lcs or not os.path.exists('params.list') or not os.path.exists('rec_results_GLSP.dat'):
		f = open("rec_list.dat", 'w')
		f.write('%d\n'%(nlc))
		for simulation in range(nlc):
			#nobs = np.random.random() * (nmax - nmin) + nmin
			#freq = np.random.random() * (freqmax - freqmin) + freqmin 
			sigm = np.random.random() * (sigmax - sigmin) + sigmin
			#params['N'] = nobs
			#params['freq'] = freq
			params['sigma'] = sigm
			x, y = get_signal(params)
			fname = 'lcs/lc_N%d_sig%.3e_freq%.3e.dat'%(params['N'], params['sigma'], params['freq'])
			save_signal(x, y, lcfile=fname)
			f.write('%s\n'%fname)

			all_pars[fname] = { key : value for key, value in params.iteritems() }
		f.close()
		pickle.dump(all_pars, open('params.list', 'wb'))
		command = "./cunfftlsf --list-in=rec_list.dat --list-out=rec_results_GLSP.dat --save-maxp --dont-save-lsp --over=%e --hifac=%e --nbootstraps=10 --nthreads=4 --memory-per-thread=128 --floating-mean"%(params['over'], params['hifac'])
		print command
		os.system(command)

		command = "./cunfftlsf --list-in=rec_list.dat --list-out=rec_results_LSP.dat  --save-maxp --dont-save-lsp --over=%e --hifac=%e --nbootstraps=10 --nthreads=12 --memory-per-thread=64"%(params['over'], params['hifac'])
		print command
		os.system(command)
	else:
		all_pars = pickle.load(open('params.list', 'rb'))
	
	

	glsp_res = load_maxp_file('rec_results_GLSP.dat')
	lsp_res = load_maxp_file('rec_results_LSP.dat')
	fnames = lsp_res['lcfname']

	lsp_res_dict = { r['lcfname'] : r for r in lsp_res}
	glsp_res_dict = { r['lcfname'] : r for r in glsp_res}

	phi_lsp =  [ (lsp_res_dict[fname]['freq'] - all_pars[fname]['freq']) / (all_pars[fname]['freq']) for fname in fnames ]
	phi_glsp =  [ (glsp_res_dict[fname]['freq'] - all_pars[fname]['freq']) / (all_pars[fname]['freq']) for fname in fnames ]

	f_lsp = [ lsp_res_dict[fname]['freq'] for fname in fnames ]
	f_glsp = [ glsp_res_dict[fname]['freq'] for fname in fnames ]

	f, ax = plt.subplots()
	ax.scatter(phi_lsp, lsp_res['fap'], alpha=0.5, color='k', marker=',', label="LSP")
	ax.scatter(phi_glsp, glsp_res['fap'], alpha=0.5, color='r', marker=',', label="GLSP")
	ax.legend(loc='best')
	#ax.set_yscale('log')
	#ax.set_xscale('log')
	ax.set_xlabel("$f/f_0 - 1$")
	ax.set_ylabel("False alarm prob")
	plt.show()

	f, ax = plt.subplots()
	ax.scatter(f_lsp, f_glsp, alpha=0.5, color='k', marker=',')
	ax.set_xlabel("f (LSP)")
	ax.set_ylabel("f (GLSP)")
	plt.show()
	'''
	f, ax = plt.subplots()
	nobs = [ all_pars[fname]['N'] for fname in fnames ]
	ax.scatter(nobs, np.abs(phi_lsp), alpha=0.5, color='k', marker=',', label="LSP")
	ax.scatter(nobs, np.abs(phi_glsp), alpha=0.5, color='r', marker=',', label="GLSP")
	ax.legend(loc='best')
	#ax.set_yscale('log')
	#ax.set_xscale('log')
	ax.set_xlabel("N observations")
	ax.set_ylabel("$f/f_0  - 1$")
	plt.show()

	f, ax = plt.subplots()
	n_lsp, r_lsp, e_lsp = get_frecovered_as_function_of(lsp_res, all_pars, 'N')
	n_glsp, r_glsp, e_glsp = get_frecovered_as_function_of(glsp_res, all_pars, 'N')
	ax.plot(n_lsp, r_lsp, lw=2, color = 'b', label="LSP")
	ax.plot(n_glsp, r_glsp, lw=2, color='r', label="GLSP")
	ax.fill_between(n_lsp, r_lsp - 0.5 * e_lsp, r_lsp + 0.5 * e_lsp, facecolor='b', alpha=0.5)
	ax.fill_between(n_glsp, r_glsp - 0.5 * e_glsp, r_glsp + 0.5 * e_glsp, facecolor='r', alpha=0.5)
	ax.legend(loc='best')
	ax.set_xlabel("N observations")
	ax.set_ylabel("frac. recovered")
	plt.show()
	'''
	f, ax = plt.subplots()
	s_lsp, r_lsp, e_lsp = get_frecovered_as_function_of(lsp_res, all_pars, 'sigma')
	s_glsp, r_glsp, e_glsp = get_frecovered_as_function_of(glsp_res, all_pars, 'sigma')
	ax.plot(s_lsp, r_lsp, lw=2, color = 'b', label="LSP")
	ax.plot(s_glsp, r_glsp, lw=2, color='r', label="GLSP")
	ax.fill_between(s_lsp, r_lsp - 0.5 * e_lsp, r_lsp + 0.5 * e_lsp, facecolor='b', alpha=0.5)
	ax.fill_between(s_glsp, r_glsp - 0.5 * e_glsp, r_glsp + 0.5 * e_glsp, facecolor='r', alpha=0.5)
	ax.legend(loc='best')
	ax.set_xlabel("S/N ratio")
	ax.set_ylabel("frac. recovered")
	plt.show()

	#for sim, pars in enumerate(all_pars):



#make_fake_lcs(Params, 1000)
inject_and_recover(Params, 1000, do_lcs=True)
'''
print "TESTING SINGLE AND DOUBLE PRECISION"
test_single_double(Params)

print "TESTING LIST OF FILES"
Params['binary'] = 'cunfftlsf'
test_list_of_files(Params)

print "TESTING FLOATING MEAN"
Params['binary'] = 'cunfftlsf'
test_floating_mean(Params)
'''
#print "TESTING GAP"
#test_gap(Params)
#freqs = np.logspace(0, 2, 10)
#diffs = test(Params, freqs)
#print diffs
#reldiffs = [ dF / F0 for F0,dF in zip(freqs, diffs) ]
#print reldiffs
#print np.mean(reldiffs), np.std(reldiffs), 
#print [ d * Params['over'] for d in diffs ]
 
#test_single_double(Params)

