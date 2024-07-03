import scipy as sc
from scipy import integrate
from math import *


f = lambda x,a : a*x

y = lambda a: integrate.quad(f, 0, 1, args=(a,))
print(y(2))

print(str(1))