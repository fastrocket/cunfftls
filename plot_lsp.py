import matplotlib.pyplot as plt
import numpy as np
from math import *
import sys, os

path = sys.argv[1]

if not (os.path.exists(path)):
	raise Exception("no filename %s exists"%(path))

lsp = np.loadtxt(path, dtype=np.dtype([ ('f', float), ('p', float)]))


plt.plot(lsp['f'], lsp['p'])
plt.show()
