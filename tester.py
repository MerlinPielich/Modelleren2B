import numpy as np
from scipy import optimize
import csv
import decimal
import math 
import scipy
import scipy.optimize as optimize
import scipy.integrate as integrate
import numpy as np
import sympy as sp

# (Radian) frequency: ?
sigma = (2*np.pi)/5.7

# Wavelength: 33.8
lamda = 33.8

#

# Wavenumber: 2pi/wavelength lambda
kappa = 2 * np.pi / lamda

# Wave amplitude: during storm maximal wave height 1.5
a = 1.5

# Water pressure: 1025 kg/m^2
rho = 1025

# Inertia coefficient: 1+C_a ~ 8.66
C_m = 8.66

# Drag coefficient: 1.17
C_D = 1.17

# Cylinder diameter: ?
D = 5

#Cilinder surface intersection
A = math.pi * (D/2)**2

# Horizontal bottom: < 50
H = 50

# z positions put into a vector z
step = .1
z = np.arange(-a, a + .00001, step)
# print(z)

# t time steps
t = np.arange(0,61,1)
# print(t)

# l Length of the windmill 150
l = 150




# variables used
# lambda_n, l, k, H = sp.symbols('lambda_n l k H')
rho = 1000
k=1/10
A = np.pi * rho * C_m * D **2 
B = 1/2 * rho * C_D * D 
lambda_n = 0.012500693792244418


### INERTIA ####
# variables used
z = sp.symbols('z')
# Equation of Z on some lambda_n
Z_eq = sp.cos(lambda_n*z) - sp.cosh(lambda_n*z) - (sp.cos(lambda_n*l) - sp.cosh(lambda_n*l))*(sp.sin(lambda_n*z) - sp.sinh(lambda_n*z))/(sp.sin(lambda_n*l) - sp.sinh(lambda_n*l))

# u, the velocity vector
u = sp.cosh(k*(z + H)) 
u_Inertia = u * Z_eq

# Integrate the u_inertia. It remains constant for t
D_inertia_const = sp.integrate(u_Inertia,(z,0,H))

t = sp.symbols('t')

D_inertia = A*(sp.sin(sigma*t)/sp.sinh(k*H))*D_inertia_const

### DRAG ###
# variables used
#same
u_Drag = u**2 * Z_eq

# Integrate the u_drag. It remains constant for t
D_Drag_const = sp.integrate(u_Drag,(z,0,H))

#The formula for the Drag part of Q
D_Drag = B*(sigma * a * sp.cos(sigma*t)/sp.sinh(k*H))**2 * D_Drag_const

Q_lamb = D_inertia + D_Drag

w_n = 1 # TEMP
tau = sp.symbols('tau')
a_inside = sp.Subs( Q_lamb,t,tau) * sp.sin(w_n*(t-tau))

a_n = sp.integrate(a_inside,(tau,0,H))

# print(float(a_n.subs(t,1)))
print(a_n)
# def Z_n( lambda_n:float = 1.0, C_n: float = 1.0,*args):
#     z = sp.symbols('z')
#     term1 = sp.cos(lambda_n * z) - sp.cosh(lambda_n * z)
#     term2 = (sp.cos(lambda_n * l) - sp.cosh(lambda_n * l)) / (sp.sin(lambda_n * l) - sp.sinh(lambda_n * l))
#     term3 = sp.sin(lambda_n * z) - sp.sinh(lambda_n * z)
#     return C_n * (term1 - term2 * term3)

# print(sp.simplify(Z_n()))

