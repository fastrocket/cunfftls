import matplotlib.pyplot as plt
import numpy as np
from math import *
import sys, os

paths = sys.argv[1:]

maxf = None
minf = None
f, ax = plt.subplots()
for path in paths:
	if not os.path.exists(path):
		raise Exception("no filename %s exists"%(path))

	lsp = np.loadtxt(path, dtype=np.dtype([ ('f', float), ('p', float)]))
	
	if '_rev' in path: 
		f = np.power(lsp['f'], -1)[::-1]
		p = lsp['p'][::-1]
		

	else:
		f = lsp['f']
		p = lsp['p']


	if maxf is None or max(f) < maxf: maxf = max(f)
	if minf is None or min(f) > minf: minf = min(f)

	ax.plot(f, p, label=path, alpha=0.5)

ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_xlim(0, maxf)#minf, maxf)
ax.legend(loc='best')
plt.show()
