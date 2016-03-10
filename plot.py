import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys


hifac = 4
over = 50
period = 0.1
N = 1000
sigma = 0.1

x = np.sort(np.random.random(N))
y = np.cos(2 * np.pi * x / period) + sigma * np.random.normal(N)

f = open("testlc.dat", 'w')
f.write("%d\n"%N)
for X, Y in zip(x, y):
	f.write("%-10e %-10e\n"%(X, Y))
f.close()

os.system("./test-double f testlc.dat testlsp-double.dat %f %f && ./test-single f testlc.dat testlsp-single.dat %f %f"%(over, hifac, over, hifac))

lspdt = np.dtype([ ('f', float), ('p', float)])

data = np.loadtxt('testlc.dat', dtype=np.dtype([ ('tobs', float), ('yobs', float) ]), skiprows=1)

single_lsp = np.loadtxt('testlsp-single.dat', dtype=lspdt)
double_lsp = np.loadtxt('testlsp-double.dat', dtype=lspdt)


fraw, axr = plt.subplots()
axr.scatter(data['tobs']/period - np.array([ int(t/period) for t in data['tobs'] ]), data['yobs'], alpha=0.1, marker='.')

fraw.savefig('lc.png')

f, ax = plt.subplots()
ax.plot(single_lsp['f'], single_lsp['p'], label='single', color='r', ls=':', alpha=0.5)
ax.plot(double_lsp['f'], double_lsp['p'], label='double', color='k', alpha=0.5)
ax.legend(loc='best')
#ax.set_xscale('log')
d = 0.1
ax.set_xlim((1./period ) * ( 1 - d ), (1./period) * (1 + d))
ax.set_ylim(0, 1.5 * max(double_lsp['p']))
ax.axvline(1./period, color='b', ls='--')

f.savefig('compare.png')

