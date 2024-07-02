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
# zeros = find_roots(f, 100, 0, 10)
# print(zeros)

def test_func(x):
    return x ** 5


# Application of simpson's rule to find the integral of a function.
def simps_rule(func, a, b, n=100):
    h = (b-a)/n
    step = a
    res = 0
    for i in range(n):
        step = a + i * h
        res += np.abs(func(step) + 4*func((step + step + h)/2) + func(step + h))
    res = (h/6) * res
    return res

def simps_rule_lambda_n(func, a, b,lambda_n, n=100,*args):
    h = (b-a)/n
    step = a
    res = 0
    for i in range(n):
        step = a + i * h
        res += np.abs(func(step, lambda_n, *args) + 4*func((step + step + h)/2,lambda_n, *args) + func(step + h,lambda_n, *args))
#        print("res simps_rule_lambda_n = ", res)
    res = (h/6) * res
    return res

def simps_rule_lambda_n_t(func, a, b,lambda_n, t, n=100,*args):
#    print("b = ",b, " a = ", a, " lambda_n = ", lambda_n, " t = ", t, " n = ", n)
    h = (b-a)/n
#    print(h)
    step = a
    res = 0
    for i in range(n):
        step = a + i * h
        res += np.abs(func(step, t, lambda_n, *args) + 4*func((step + step + h)/2, t, lambda_n, *args) + func(step + h, t, lambda_n, *args))
#        print("res simps_rule_lambda_n_t = ", res)
    res = (h/6) * res
    #print("res = ", res)
    return res

# # Z_n: The space dependent part of the SOV of the beam equation
# def Z_n(x: float, lambda_n:float = 1.0, C_n: float = 1.0,*args):
#     term1 = np.cos(lambda_n * x) - np.cosh(lambda_n * x)
#     term2 = (np.cos(lambda_n * l) - np.cosh(lambda_n * l)) / (np.sin(lambda_n * l) - np.sinh(lambda_n * l))
#     term3 = np.sin(lambda_n * x) - np.sinh(lambda_n * x)
#     return C_n * (term1 - term2 * term3)

#print(Z_n(1))

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



def func1(x,t,lambda_n,rho = rho[0],*args):
    return wave_force(x, rho,t)*Z_n(x,lambda_n)

def Q_n(t:float, lambda_n, *args):
    return simps_rule_lambda_n_t(func1, 0, l, lambda_n, t)

def func2(tau, t,lambda_n:float,*args):
#    print("func2 left part: Q_n(tau, lambda_n) = ", Q_n(tau, lambda_n))
#    print("func2 right part: ", np.sin(lambda_n*(t-tau)))
    return Q_n(tau, lambda_n)*np.sin(lambda_n*(t-tau))

def func_b(Z_n,lambda_n,*args):
    return simps_rule_lambda_n(Z_n_sq, a=0, b=H, lambda_n=lambda_n)

def small_q(t_step,lambda_n,*args):
    q_n = (1/(rho[0]*A*func_b(Z_n,lambda_n)*lambda_n))*simps_rule_lambda_n_t(func2, a=0, b=t_step, lambda_n=lambda_n, t=t_step)
#    print("left part small_q", (1/(rho[0]*A*func_b(Z_n,lambda_n)*lambda_n)))
#    print("right part small_q", simps_rule_lambda_n_t(func2, a=0, b=t_step, lambda_n=lambda_n, t=t_step))
    return q_n

# def a_t():

    

#     expression = C*(np.cos(lambda_n*z)*np.sin(lambda_n*l) - np.cos(lambda_n*z)*np.sinh(lambda_n*l) 
#                 - np.cosh(lambda_n*z)*np.sin(lambda_n*l) + np.cosh(lambda_n*z)*np.sinh(lambda_n*l) 
#                 - np.cos(lambda_n*l)*np.sin(lambda_n*z) + np.cos(lambda_n*l)*np.sinh(lambda_n*z) 
#                 + np.cosh(lambda_n*l)*np.sin(lambda_n*z) 
#                 - np.cosh(lambda_n*l)*np.sinh(lambda_n*z))*np.cosh(k(H + z))/(np.sin(lambda_n*l) 
#                 - np.sinh(lambda_n*l))

#print(small_q(1, 1))

def w(x,t,lambda_list:list,*args):
    w = 0
    n = len(lambda_list)
    count = 1
    for lambda_n in lambda_list:
        w += Z_n(x=x,lambda_n=lambda_n) * small_q(t_step=t,lambda_n = lambda_n)
#        print("Z_n = ", Z_n(x=x, lambda_n=lambda_n))
#        print("small q = ", small_q(t_step=t,lambda_n=lambda_n))
#        print("w = ",w)
#        print("w lambda_n number = ", count)
        count += 1
#    print("w =", w)
    return w

# Z_n: The space dependent part of the SOV of the beam equation
def Z_n(lambda_n:float = 1.0, C_n: float = 1.0,*args):
    z = sp.symbols('z')
    term1 = sp.cos(lambda_n * z) - sp.cosh(lambda_n * z)
    term2 = (sp.cos(lambda_n * l) - sp.cosh(lambda_n * l)) / (sp.sin(lambda_n * l) - sp.sinh(lambda_n * l))
    term3 = sp.sin(lambda_n * z) - sp.sinh(lambda_n * z)
    return C_n * (term1 - term2 * term3)


def BEQ(t_end:float = 2,dt:float = 1,l:int = 150,dl:float = 10):
    lambda_list = find_lambdas(frequency_equation, 100, 0, .1)
    print("amount of lambdas = ",len(lambda_list))
    n = len(lambda_list)
    heights = np.arange(0,l,dl)
    z = []
    z_new = 0
    t_step = 0
    Z_h = 0
    Z_n_list = []
    a_t = 0
    U = []

    #Calculate all the values for the space dependent parts 
    for lambda_n in lambda_list:
        Z_n_list.append(Z_n(lambda_n = lambda_n))

    print(f"the first value for Z_n is {Z_n_list[0]}")


    #Calculate prelimenary integrals for the space depentent parts of Q
    a_n = []
    for lambda_n in lambda_list:

        A = np.pi * rho * C_m * D **2 
        B = 1/2 * rho * C_D * D 

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

        # The formula for the Drag part of Q
        D_Drag = B*(sigma * a * sp.cos(sigma*t)/sp.sinh(k*H))**2 * D_Drag_const

        Q_lamb = D_inertia + D_Drag


        # The time depentent part of the decomposition. The entries in a_n (list)
        # are sympy objects with input t
        w_n = 1 # TEMP
        tau = sp.symbols('tau')
        a_const = 1 / (rho  * A * 1) #TODO!!!
        a_inside = a_const * sp.Subs( Q_lamb,t,tau) * sp.sin(w_n*(t-tau))

        
        a_n.append(a_inside)

    print(f"the first a_n is {a_n[0]}")
    #structure of output
    #[ [time1,[ [x1,z1], [x2,z2],...  ]],[time2,[ [x1,z1], [x2,z2],...  ]],...  ]

    while t_step <= t_end:
        U_t = []

        print("t_step = ", t_step)
        U_z = 0

        for count,lambda_n in enumerate(lambda_list):
            # compute formula of Deviation U at time t
            # U_z is dependent of its height: z
            a_lambda_t = a_n[count]
            a_lambda = float(sp.integrate(a_lambda_t,(t,0,t_step)))

            Z_lambda_z = Z_n_list[count]

            U_z += a_lambda * Z_lambda_z
        print(f"U_z is {U_z}")

        for h,count in enumerate(heights):
            U_t.append([h,sp.substitution(U_z,z,h)])

        U.append([t , U_t])
    
    return U


bop = BEQ()
print(bop)
            




# z = BEQ()
# filename = "results.csv"
# with open(filename, 'w') as csvfile:
#     # creating a csv dict writer object
#     writer = csv.DictWriter(csvfile)
 
#     # writing data rows
#     writer.writerows(z)
            
    


# test, answer should be 12
# print(simps_rule(test_func, 0, 2, 1))




# print(find_lambdas(frequency_equation, 10, 0, .1))








