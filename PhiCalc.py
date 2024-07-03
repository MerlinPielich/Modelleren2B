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
import scipy as sc
from scipy import integrate
from itertools import accumulate
# from sympy.utilities.lambdify import lambdify, implemented_function
from operator import add
#Helpt dit?
def function_operation(h, f, g): return lambda z: h(f(z), g(z))
def scalar_multiplication(s,f): return lambda z: s * f(z)

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

# # Z_n: The npace dependent part of the SOV of the beam equation
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

# Z_n: The npace dependent part of the SOV of the beam equation
def Z_n(lambda_n:float = 1.0, C_n: float = 1.0,*args):
    
    return lambda z,C_n = C_n, lambda_n =lambda_n: C_n *( np.cos(lambda_n * z) - np.cosh(lambda_n * z) \
        - (np.cos(lambda_n * l) - np.cosh(lambda_n * l)) / (np.sin(lambda_n * l) - np.sinh(lambda_n * l)) \
        * np.sin(lambda_n * z) - np.sinh(lambda_n * z) )

def Q_n(lambda_n, A, B):

    k = kappa
    # Inertia
    Z_eq = lambda z: np.cos(lambda_n*z) - np.cosh(lambda_n*z) - (np.cos(lambda_n*l) - np.cosh(lambda_n*l))*(np.sin(lambda_n*z) - np.sinh(lambda_n*z))/(np.sin(lambda_n*l) - np.sinh(lambda_n*l))
    u = lambda z: np.cosh(kappa*(z + H)) 
    u_Inertia = lambda z: math.prod( [ u(z) , Z_eq(z) ] )
    # print(f"formula for u_inertia is given by {print(u_Inertia)}")

    # Integrate the u_inertia. It remains constant for t
    D_inertia_const = integrate.quad(u_Inertia, a = 0, b = 50) 


    # print(f"The value for D_i_c is  {D_inertia_const}")


    D_inertia = lambda t: A*(np.sin(sigma*t)/np.sinh(k*H))*D_inertia_const[0]
    # print(f"D_inertia formula is given by {D_inertia}")

    ### DRAG ###
    # variables used
    #same
    u_Drag = lambda z: math.prod( [ u(z)**2, Z_eq(z) ] )

    # Integrate the u_drag. It remains constant for t
    D_Drag_const = integrate.quad(u_Drag,0,H)
    D_Drag = lambda t: B*(sigma * a * np.cos(sigma*t)/np.sinh(k*H))**2 * D_Drag_const[0]

    return lambda t: D_inertia(t) + D_Drag(t)

def a_n(w_n,a_const,n,Q_n):
    return lambda t,n=n:  a_const*( \
          np.sin(w_n*t)   * integrate.quad(lambda tau: Q_n(tau) * np.cos(w_n * tau), a = 0, b = t)[0] \
        - np.cos(w_n*t)   * integrate.quad(lambda tau: Q_n(tau) * np.sin(w_n * tau), a = 0, b = t)[0]) 
    
def BEQ(t_end:float = 2,dt:float = 1,l:int = 150,dl:float = 50):
    
    lambda_list = find_lambdas(frequency_equation, 100, 0, 0.5)
    
    print("amount of lambdas = ",len(lambda_list))
    
    n = len(lambda_list)
    
    heights = np.arange(0,l,dl)
    
    z = []
    z_new = 0
    t_step = 1
    Z_h = 0
    Z_n_list = []
    a_t = 0
    U = []
    k = kappa
    a_n_list = []

    #Calculate all the values for the npace dependent parts 
    for count,lambda_n in enumerate(lambda_list):
        Z_n_list.append(lambda z: Z_n(lambda_n = lambda_n))
        w_n = 1
        A = np.pi * rho[0] * C_m * D **2 
        B =   1/2 * rho[0] * C_D * D 

        # Q_list.append(Q_n(lambda_n=lambda_n,A = A,B = B))

        a_const = 1 / (rho[0]  * A * 1) #TODO!!!
        # a_n.append( lambda tau,t: a_const * Q_lamb(tau) * np.sin(w_n*(t-tau)) )
        a_n_list.append(a_n(a_const=a_const,w_n=w_n, n=count, Q_n=Q_n(lambda_n=lambda_n,A = A,B = B)) )



    print(f"the first value for Z_n is {Z_n_list[0](150)}")



    # print(f"the first a_n is {print(a_n[0])}")
    #structure of output
    #[ [time1,[ [x1,z1], [x2,z2],...  ]],[time2,[ [x1,z1], [x2,z2],...  ]],...  ]

    while t_step <= t_end:
        U_t = []

        print("t_step = ", t_step)
        
        U_z = lambda z: 0
        a_lambda_list = []
        Z = []
        
        print(f"the list of lambda's is {lambda_list}")
        
        for count,lambda_n in enumerate(lambda_list):
            # compute formula of Deviation U at time t
            # U_z is dependent of its height: z
            # a_lambda_t = lambda tau,t: (a_n[count])(tau,t)
            # print(f"count is given by number {count}")
            
            
            
            # a_lambda, a_lambda_error =  integrate.quad(a_n[count], a = 0, b = t_step, args=(t_step,))
            
            a_lambda = (a_n_list[count])(t_step,count) 
            
            print(f"a_lambda is {a_lambda} ")

            # Z.append(  implemented_function(f'{count}', lambda z: a_lambda * Z_n_list[count] ) )
            Z_lambda = (Z_n_list[count])(z)
            W_n = lambda z: a_lambda * Z_lambda(z)
                    
            U_z = function_operation(add, W_n, U_z)
            
        # U_z = lambda z: accumulate(Z, operator.add)
        # lambda z: a_lambda[0] * Z_lambda_z(z) + U_z(z)
        # print(f"U_z at the top of the beam is is {U_z(H)}")

        
        U.append([t_step,[U_z(h) for h in heights]])    

        # U.append([t_step , U_t])
        
        t_step += dt
    # print(f"THe testlist states: {testlist}")
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








