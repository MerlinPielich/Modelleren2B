# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:52:55 2024

@author: Maris
"""
import numpy as np
from scipy import optimize
import csv
import decimal
import math 

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

# Function that calculates phi
def calc_phi(z, x, t):
    phi = sigma / kappa * a * np.cosh(kappa * (z+H)) / \
        np.sinh(kappa*H) * np.sin((kappa * x)+(sigma * t))
    return phi


# Function that calculates the differential of phi
def diff_phi(z, x, t):
    d_phi = sigma * kappa * a * np.cosh(kappa * (z + H)) / \
        np.sinh(kappa * H) * np.cos((kappa * x) + (sigma * t))
    return d_phi


# Wave force function/morrison equation. Use this one to calculate wave forcing
def wave_force(z, rho, t):
    F = (np.pi / 4 * rho * C_m * D ** 2 * diff_u(z, t)) + \
        1/2 * rho * C_D * D * u(z, t) * np.abs(u(z, t))
    return F

# wave force function
def wave_force_var_dens(z, rho, t):
    F = [[] for i in range(len(rho))]
    for j in range(len(rho)):
        F[j] = (np.pi / 4 * rho[j] * C_m * D ** 2 * diff_u(z, t)) + \
        1/2 * rho[j] * C_D * D * u(z, t) * np.abs(u(z, t))
    return F

# Function that calculates u. This is simply a function that is used in the wave forcing function
def u(z, t):
    u = sigma * a * np.cosh(kappa * (z + H)) / \
        np.sinh(kappa * H) * np.cos(sigma * t)
    return u


# function for the differential of u in z. This is simply a function that is used in the wave forcing function
def diff_u(z, t):
    diff_u = - sigma ** 2 * a * np.cosh(kappa * (z + H)) / \
        np.sinh(kappa*H) * np.sin(sigma * t)
    return diff_u


# find force at wave position z at time t.
force = np.empty((len(t),len(z)))
for k in range(len(t)):
    for i in range(len(z)):
        force[k][i] = wave_force(i,rho,k)

end = 1500
step2 = 1
rho = np.arange(1025, end+1, step2)

force_var_dens = np.empty((len(t), len(z), len(rho)))
for k in range(len(t)):
    for i in range(len(z)):
        for y in range(len(rho)):
            force_var_dens[k][i] = wave_force(i,rho[y],k)

# Function of which the zeros have to be found.
def f(x):
    return np.cosh(x)*np.cos(x)+1


# Apply bisection method to find the zeros of a function.
def find_roots(func, n, a, b):
    zeros = [[[], [], []]for i in range(n)]
    stepsize = (b-a)/n
    for i in range(n):
        A = a + stepsize*i
        B = a + stepsize * (i+1)
        zeros[i][0] = A
        zeros[i][1] = B
        if func(A) == 0 or func(B) == 0 \
                or (func(A) < 0 and func(B) > 0) \
                or (func(B) < 0 and func(A) > 0):
            zeros[i][2] = optimize.bisect(func, A, B)
        else:
            zeros[i][2] = None
    return zeros


def test_q(x):
    return np.sin(x)


# lambda's for the beam equation
zeros = find_roots(f, 100, 0, 100)


def test_func(x):
    return x ** 5


# Application of simpson's rule to find the integral of a function.
def simps_rule(func, a, b, n=100):
    h = (b-a)/n
    step = a
    res = 0
    for i in range(n):
        step = a + i * h
        res += func(step) + 4*func((step + step + h)/2) + func(step + h)
    res = (h/6) * res
    return res

def simps_rule_lambda_n(func, a, b,lambda_n, n=100,*args):
    h = (b-a)/n
    step = a
    res = 0
    print("initialised res = ", res)
    return
    for i in range(n):
        step = a + i * h
        res += func(step, lambda_n, *args) + 4*func((step + step + h)/2,lambda_n, *args) + func(step + h,lambda_n, *args)
        print("res simps_rule_lambda_n = ", res)
    res = (h/6) * res
    return res

def simps_rule_lambda_n_t(func, a, b,lambda_n, t, n=100,*args):
    h = (b-a)/n
    step = a
    res = 0
    for i in range(n):
        step = a + i * h
        res += func(step, t, lambda_n, *args) + 4*func((step + step + h)/2, t, lambda_n, *args) + func(step + h, t, lambda_n, *args)
        print("res simps_rule_lambda_n_t = ", res)
    res = (h/6) * res
    return res

# Z_n: The space dependent part of the SOV of the beam equation
def Z_n(x: float, lambda_n:float = 1.0, C_n: float = 1.0,*args):
    term1 = np.cos(lambda_n * x) - np.cosh(lambda_n * x)
    term2 = (np.cos(lambda_n * l) - np.cosh(lambda_n * l)) / (np.sin(lambda_n * l) - np.sinh(lambda_n * l))
    term3 = np.sin(lambda_n * x) - np.sinh(lambda_n * x)
    return C_n * (term1 - term2 * term3)

print(Z_n(1))

def Z_n_sq(x: float, lambda_n:float = 1.0, C_n: float = 1.0,*args):
    term1 = np.cos(lambda_n * x) - np.cosh(lambda_n * x)
    term2 = (np.cos(lambda_n * l) - np.cosh(lambda_n * l)) / (np.sin(lambda_n * l) - np.sinh(lambda_n * l))
    term3 = np.sin(lambda_n * x) - np.sinh(lambda_n * x)
    return (C_n * (term1 - term2 * term3))**2

def frequency_equation(x:float,l:float = 150.0) -> float:
    return np.cosh(x*l)*np.cos(x*l)+1

def find_lambdas(func, n, a, b):
    zeros = []
    stepsize = (b-a)/n
    for i in range(n):
        A = a + stepsize*i
        B = a + stepsize * (i+1)
        if func(A)*func(B) < 0:
            zeros.append(optimize.bisect(func, A, B))
    return zeros

##print(len(find_lambdas(frequency_equation, 1000, 0, 2)))



def func1(x,t,lambda_n,rho = rho,*args):
    return wave_force(x, rho,t)*Z_n(x,lambda_n)

def Q_n(t:float, lambda_n, *args):
    return simps_rule_lambda_n_t(func1, 0, l, lambda_n, t)

def func2(tau, t,lambda_n:float,*args):
    return Q_n(tau, lambda_n)*np.sin(lambda_n*(t-tau))

def func_b(Z_n,lambda_n,*args):
    return simps_rule_lambda_n(Z_n_sq, a=0, b=H, lambda_n=lambda_n)

def small_q(t,lambda_n,*args):
    q_n = (1/(rho*A*func_b(Z_n,lambda_n)*lambda_n))*simps_rule_lambda_n_t(func2, a=0, b=t, lambda_n=lambda_n, t=t)
    return q_n

print(small_q(1, 1))

def w(x,t,lambda_list:list,*args):
    w = 0
    n = len(lambda_list)
    count = 1
    for lambda_n in lambda_list:
        w += Z_n(x=x,lambda_n=lambda_n) * small_q(t=t,lambda_n = lambda_n)
        print("w = ",w)
        print("w lambda_n number = ", count)
        count += 1
    print("w =", w)
    return w


def BEQ(t_end:float = 30,dt:float = 1,l:int = 150,dl:float = 1.0):
    lambda_list = find_lambdas(frequency_equation, 1000, 0, 1)
    print(len(lambda_list))
    n = len(lambda_list)
    Z = []
    z_new = 0
    t = 0

    #structure of output
    #[ [time1,[ [x1,z1], [x2,z2],...  ]],[time2,[ [x1,z1], [x2,z2],...  ]],...  ]

    while t <= t_end:
        print(t)
        z_x = []

        for x in np.arange(0,l,dl):
            #main block
            print("x = ", x )


            #this part is extremely inefficient, gotta look at this later.
            #Right now it calculates w for every x and t whilst
            #this is only required for a specific t.
            z_new = w(x,t,lambda_list)
            print("z_new done")
            #z_x is a list with the w for different points on the beam
            z_x.append([x,z_new])

            #end block: time step
        t += dt

    
        z_t = [t,z_x]
        print(z_t)
            
        z.append(z_t)

    return(z)


z = BEQ()
filename = "results.csv"
with open(filename, 'w') as csvfile:
    # creating a csv dict writer object
    writer = csv.DictWriter(csvfile)
 
    # writing data rows
    writer.writerows(z)
            
    
    
    

# test, answer should be 12
# print(simps_rule(test_func, 0, 2, 1))











