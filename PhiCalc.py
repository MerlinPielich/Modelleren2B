# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:52:55 2024

@author: Maris en Merli :P
"""
import csv
import decimal
import math
from functools import cache
from itertools import accumulate
# from sympy.utilities.lambdify import lambdify, implemented_function
from operator import add

import numpy as np
import scipy
import scipy as sc
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy import integrate, optimize


# Helpt dit?
def function_operation(h, f, g):
    return lambda z: h(f(z), g(z))


def scalar_multiplication(s, f):
    return lambda z: s * f(z)


def multiplication(f, g):
    return lambda z: math.prod([func(z) for func in [f, g]])


# (Radian) frequency: ?
sigma = (2 * np.pi) / 5.7

# Wavelength: 33.8
lamda = 33.8

# I
I = 0.003067962
# E
E = 200 * 10**9
# EI
EI = E * I

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

# Cilinder surface intersection
A = math.pi * (D / 2) ** 2

# Horizontal bottom: < 50
H = 50

# z positions put into a vector z
step = 0.1
z = np.arange(-a, a + 0.00001, step)
# print(z)

# t time steps
t = np.arange(0, 61, 1)
# print(t)

# l Length of the windmill 150
l = 150

# density of steel
rho_steel = 7850


# Function that calculates phi
@cache
def calc_phi(z, x, t):
    phi = (
        sigma
        / kappa
        * a
        * np.cosh(kappa * (z + H))
        / np.sinh(kappa * H)
        * np.sin((kappa * x) + (sigma * t))
    )
    return phi


# Function that calculates the differential of phi
@cache
def diff_phi(z, x, t):
    d_phi = (
        sigma
        * kappa
        * a
        * np.cosh(kappa * (z + H))
        / np.sinh(kappa * H)
        * np.cos((kappa * x) + (sigma * t))
    )
    return d_phi


# Wave force function/morrison equation. Use this one to calculate wave forcing
@cache
def wave_force(z, rho, t):
    F = (np.pi / 4 * rho * C_m * D**2 * diff_u(z, t)) + 1 / 2 * rho * C_D * D * u(
        z, t
    ) * np.abs(u(z, t))
    return F


# wave force function
@cache
def wave_force_var_dens(z, rho, t):
    F = [[] for i in range(len(rho))]
    for j in range(len(rho)):
        F[j] = (np.pi / 4 * rho[j] * C_m * D**2 * diff_u(z, t)) + 1 / 2 * rho[
            j
        ] * C_D * D * u(z, t) * np.abs(u(z, t))
    return F


# Function that calculates u. This is simply a function that is used in the wave forcing function
def u(z, t):
    u = sigma * a * np.cosh(kappa * (z + H)) / np.sinh(kappa * H) * np.cos(sigma * t)
    return u


# function for the differential of u in z. This is simply a function that is used in the wave forcing function
def diff_u(z, t):
    diff_u = (
        -(sigma**2)
        * a
        * np.cosh(kappa * (z + H))
        / np.sinh(kappa * H)
        * np.sin(sigma * t)
    )
    return diff_u


# find force at wave position z at time t.
force = np.empty((len(t), len(z)))
for k in range(len(t)):
    for i in range(len(z)):
        force[k][i] = wave_force(i, rho, k)

end = 1500
step2 = 1
rho = np.arange(1000, end + 1, step2)

force_var_dens = np.empty((len(t), len(z), len(rho)))
for k in range(len(t)):
    for i in range(len(z)):
        for y in range(len(rho)):
            force_var_dens[k][i] = wave_force(i, rho[y], k)


# Function of which the zeros have to be found.
@cache
def f(x):
    return np.cosh(x) * np.cos(x) + 1


# Apply bisection method to find the zeros of a function.
def find_roots(func, n, a, b):
    zeros = [[[], [], []] for i in range(n)]
    stepsize = (b - a) / n
    for i in range(n):
        A = a + stepsize * i
        B = a + stepsize * (i + 1)
        zeros[i][0] = A
        zeros[i][1] = B
        if (
            func(A) == 0
            or func(B) == 0
            or (func(A) < 0 and func(B) > 0)
            or (func(B) < 0 and func(A) > 0)
        ):
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
    return x**5


# Application of simpson's rule to find the integral of a function.
def simps_rule(func, a, b, n=100):
    h = (b - a) / n
    step = a
    res = 0
    for i in range(n):
        step = a + i * h
        res += np.abs(func(step) + 4 * func((step + step + h) / 2) + func(step + h))
    res = (h / 6) * res
    return res


@cache
def simps_rule_lambda_n(func, a, b, lambda_n, n=100, *args):
    h = (b - a) / n
    step = a
    res = 0
    for i in range(n):
        step = a + i * h
        res += np.abs(
            func(step, lambda_n, *args)
            + 4 * func((step + step + h) / 2, lambda_n, *args)
            + func(step + h, lambda_n, *args)
        )
    #        print("res simps_rule_lambda_n = ", res)
    res = (h / 6) * res
    return res


@cache
def simps_rule_lambda_n_t(func, a, b, lambda_n, t, n=100, *args):
    #    print("b = ",b, " a = ", a, " lambda_n = ", lambda_n, " t = ", t, " n = ", n)
    h = (b - a) / n
    #    print(h)
    step = a
    res = 0
    for i in range(n):
        step = a + i * h
        res += np.abs(
            func(step, t, lambda_n, *args)
            + 4 * func((step + step + h) / 2, t, lambda_n, *args)
            + func(step + h, t, lambda_n, *args)
        )
    #        print("res simps_rule_lambda_n_t = ", res)
    res = (h / 6) * res
    # print("res = ", res)
    return res


# # Z_n: The npace dependent part of the SOV of the beam equation
# def Z_n(x: float, lambda_n:float = 1.0, C_n: float = 1.0,*args):
#     term1 = np.cos(lambda_n * x) - np.cosh(lambda_n * x)
#     term2 = (np.cos(lambda_n * l) - np.cosh(lambda_n * l)) / (np.sin(lambda_n * l) - np.sinh(lambda_n * l))
#     term3 = np.sin(lambda_n * x) - np.sinh(lambda_n * x)
#     return C_n * (term1 - term2 * term3)


# print(Z_n(1))
@cache
def Z_n_sq(x: float, lambda_n: float = 1.0, C_n: float = 1.0, *args):
    term1 = np.cos(lambda_n * x) - np.cosh(lambda_n * x)
    term2 = (np.cos(lambda_n * l) - np.cosh(lambda_n * l)) / (
        np.sin(lambda_n * l) - np.sinh(lambda_n * l)
    )
    term3 = np.sin(lambda_n * x) - np.sinh(lambda_n * x)
    return (C_n * (term1 - term2 * term3)) ** 2


@cache
def frequency_equation(x: float, l: float = 150.0) -> float:
    return np.cosh(x * l) * np.cos(x * l) + 1


@cache
def find_lambdas(func, n, a, b):
    zeros = []
    stepsize = (b - a) / n
    for i in range(n):
        A = a + stepsize * i
        B = a + stepsize * (i + 1)
        if func(A) * func(B) < 0:
            zeros.append(optimize.bisect(func, A, B))
    return zeros


##print(len(find_lambdas(frequency_equation, 1000, 0, 2)))


@cache
def func1(x, t, lambda_n, rho=rho[0], *args):
    return wave_force(x, rho, t) * Z_n(x, lambda_n)


@cache
def Q_n(t: float, lambda_n, *args):
    return simps_rule_lambda_n_t(func1, 0, l, lambda_n, t)


@cache
def func2(tau, t, lambda_n: float, *args):
    #    print("func2 left part: Q_n(tau, lambda_n) = ", Q_n(tau, lambda_n))
    #    print("func2 right part: ", np.sin(lambda_n*(t-tau)))
    return Q_n(tau, lambda_n) * np.sin(lambda_n * (t - tau))


@cache
def func_b(Z_n, lambda_n, *args):
    return simps_rule_lambda_n(Z_n_sq, a=0, b=H, lambda_n=lambda_n)


@cache
def small_q(t_step, lambda_n, *args):
    q_n = (
        1 / (rho_steel * A * func_b(Z_n, lambda_n) * lambda_n)
    ) * simps_rule_lambda_n_t(func2, a=0, b=t_step, lambda_n=lambda_n, t=t_step)
    #    print("left part small_q", (1/(rho_steel*A*func_b(Z_n,lambda_n)*lambda_n)))
    #    print("right part small_q", simps_rule_lambda_n_t(func2, a=0, b=t_step, lambda_n=lambda_n, t=t_step))
    return q_n


# def a_t():


#     expression = C*(np.cos(lambda_n*z)*np.sin(lambda_n*l) - np.cos(lambda_n*z)*np.sinh(lambda_n*l)
#                 - np.cosh(lambda_n*z)*np.sin(lambda_n*l) + np.cosh(lambda_n*z)*np.sinh(lambda_n*l)
#                 - np.cos(lambda_n*l)*np.sin(lambda_n*z) + np.cos(lambda_n*l)*np.sinh(lambda_n*z)
#                 + np.cosh(lambda_n*l)*np.sin(lambda_n*z)
#                 - np.cosh(lambda_n*l)*np.sinh(lambda_n*z))*np.cosh(k(H + z))/(np.sin(lambda_n*l)
#                 - np.sinh(lambda_n*l))


# print(small_q(1, 1))
# ! not in use anymore
@cache
def w(x, t, lambda_list: list, *args):
    w = 0
    n = len(lambda_list)
    count = 1
    for lambda_n in lambda_list:
        w += Z_n(x=x, lambda_n=lambda_n) * small_q(t_step=t, lambda_n=lambda_n)
        count += 1
    #    print("w =", w)
    return w


# * The function to caculate the space dependent part of the SOV
@cache
def Z_n(lambda_n: float = 1.0, *args):
    wn = lambda_n * l

    C_n = -(np.cos(lambda_n * l) - np.cosh(lambda_n * l)) / (
        np.sin(lambda_n * l) - np.sinh(lambda_n * l)
    )

    return (
        lambda z: (1 / 2)
        * ((1 - C_n) * np.exp(lambda_n * z) + (1 + C_n) * np.exp(-lambda_n * z))
        + C_n * np.sin(lambda_n * z)
        + np.cos(lambda_n * z)
    )


@cache
def Q_n(lambda_n, A, B):

    # * introduced constants:
    k = kappa

    # * u is the function for the wave forcing. Here initialized as lambda function
    u = lambda z: (
        (np.exp(kappa * (z + H)) + np.exp(-kappa * (z + H))) / 2 if z <= 50 else 0
    )

    # * Z_eq is the function for this lambda
    Z_eq = Z_n(lambda_n=lambda_n)

    # (
    #     lambda z: np.cos(lambda_n * z)
    #     - np.cosh(lambda_n * z)
    #     - (np.cos(lambda_n * l) - np.cosh(lambda_n * l))
    #     * (np.sin(lambda_n * z) - np.sinh(lambda_n * z))
    #     / (np.sin(lambda_n * l) - np.sinh(lambda_n * l))
    # )

    # * ## Inertia Calculations ## * #

    # * Inertia part for Q
    u_Inertia = lambda z: math.prod([u(z), Z_eq(z)])

    # * Integrate the u_inertia. It remains constant for t.
    D_inertia_const = integrate.quad(u_Inertia, a=0, b=H)

    # * The inertia itself is dependent on t. So it is put in a lambda function
    D_inertia = lambda t: A * (
        np.sin(sigma * t) / ((np.exp(k * H) - np.exp(-k * H)) / 2) * D_inertia_const[0]
    )

    # * ## Drag Calculations ## * #
    u_Drag = lambda z: math.prod([u(z) ** 2, Z_eq(z)])

    # * Integrate the u_drag. It remains constant for t
    D_Drag_const = integrate.quad(u_Drag, 0, H)
    D_Drag = (
        lambda t: B
        * (sigma * a * np.cos(sigma * t) / ((np.exp(k * H) - np.exp(-k * H)) / 2)) ** 2
        * D_Drag_const[0]
    )

    # * Return the sum of both functions
    Q = function_operation(add, D_inertia, D_Drag)
    return Q


# ? Function needs to be created top level in order to not be overwritten
@cache
def a_n(mu_n, a_const, n, Q_n):
    # * Return time dependent part of the sov
    return lambda t, n=n: a_const * (
        np.sin(mu_n * t)
        * integrate.quad(lambda tau: Q_n(tau) * np.cos(mu_n * tau), a=0, b=t)[0]
        - np.cos(mu_n * t)
        * integrate.quad(lambda tau: Q_n(tau) * np.sin(mu_n * tau), a=0, b=t)[0]
    )


# * Alternative function to compute eigenvalues
def compute_evs():
    r = 0.3043077 * 1.1
    mu = rho_steel * A
    motes = 10
    L = l
    eigenvalues = np.zeros(motes)
    alfas = np.zeros(motes)
    for i in range(0, motes):
        c = np.pi * (i + 0.5)

        # Compute the root in normalized coordinates then scale back.
        yr = optimize.root_scalar(f, bracket=[c - r, c + r], method="brentq").root
        xr = yr / L
        eigenvalues[i] = xr
        alfas[i] = xr**2 * np.sqrt(E * I / mu)
    return eigenvalues


def b_list(lambda_list: list):
    b = []

    for lambda_n in lambda_list:
        Zn = Z_n(lambda_n=lambda_n)
        Zn_squared = multiplication(Zn, Zn)

        b.append(integrate.quad(Zn_squared, a=0, b=l)[0])

        # ! Error Checker
        print(f"Zn with imput 10 is {b[-1]}")
    return b


def BEQ(t_end: float = 15, dt: float = 1, l: int = 150, dl: float = 50):
    # * Constants
    A = math.pi * (D / 2) ** 2
    rho_steel = 7850

    # lambda_list = find_lambdas(frequency_equation, 100, 0, 0.1)
    lambda_list = compute_evs()
    # ! Error Checker
    print("amount of lambdas = ", len(lambda_list))
    print(f"\n The lambdas are: \n {lambda_list}")

    # * initialize heights and time_steps to be used in for loops
    heights = np.arange(0, l + dl, dl)
    time_steps = np.arange(0, t_end + dt, dt)

    # * Z_n_list of the space dependent parts of the SOV
    # * W_total is used to store the final result
    # * a_n_list stores the values for the time dependent part of th SOV
    # * b are constant values dependent on lambda
    Z_n_list = []  # list(function)
    W_total = (
        []
    )  # [ [time1,[ [x1,z1], [x2,z2],...  ]],[time2,[ [x1,z1], [x2,z2],...  ]],...  ]
    a_n_list = []  # list(floats)

    b = b_list(lambda_list)

    # * Calculate all the values for the space dependent parts and
    # * the correspoinding time dependent functions for each lambda.
    for count, lambda_n in enumerate(lambda_list):

        # * Z_n_list contains the funciton Z_n as lambda functions
        # ? Parameters: z
        Z_n_list.append(Z_n(lambda_n=lambda_n))

        # * constants:
        # * mu_n dependent on lambda,
        # * A, B constants
        # * a_const dependent on lambdas
        mu_n = (lambda_n * l) * (EI / (rho_steel * A * (l**4))) ** 4
        A_a_n = np.pi * rho_steel * C_m * D**2
        B_a_n = 1 / 2 * rho_steel * C_D * D
        a_const = 1 / (rho_steel * A * mu_n * b[count])

        # * a_n_list constains lambda functions of a_n(t). Input for the functions is t
        a_n_list.append(
            a_n(
                a_const=a_const,
                mu_n=mu_n,
                n=count,
                Q_n=Q_n(lambda_n=lambda_n, A=A_a_n, B=B_a_n),
            )
        )

    print(f"\nthe first value for Z_n is \n {(Z_n_list[0])(80)}\n")

    # ! Error checkers
    # print(f"the first a_n is {print(a_n[0])}")
    print(f"\nthe list of lambda's is \n {lambda_list}\n")

    # * structure of output for the function BEQ
    # [ [time1,[ [x1,z1], [x2,z2],...  ]],[time2,[ [x1,z1], [x2,z2],...  ]],...  ]

    for t_step in time_steps:

        # ! Error checker
        print(f"t_step ={t_step}")

        # * Initialize lambda function for the deviation
        W = lambda z: 0

        for count, lambda_n in enumerate(lambda_list):
            # * compute formula of Deviation U at time t

            # ! Error checker
            # print(f"count is given by number {count}")

            # * calculates the time dependent part of the SOV
            # * This is constant for all z so its constant for this t_step
            a_lambda = (a_n_list[count])(t_step, count)

            # * Time dependent part of SOV
            Z_lambda = lambda z: Z_n_list[count](z)

            # ! Error checker a_lambda and z_lambda
            print(f"a_lambda is {a_lambda} ")
            print(f"z_lambda(150) is {Z_lambda(150)} ")

            # ! Error checker
            # print(f"Z_n in the function for input 10 provides {(zn)(z=10)}")

            # * W_n is the multiplied of the time and space dependent parts
            # * of the SOV. W is the sum of those values. Both are stored as lambda funcitons
            W_lambda = scalar_multiplication(a_lambda, Z_lambda)
            W = function_operation(add, W_lambda, W)

        # ! Error checker
        # print(f"W at the top of the beam is is {W(H)}")

        W_total.append([t_step, [W(h) for h in heights]])

    return W_total


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
