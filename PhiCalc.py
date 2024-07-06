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

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy as sc
import scipy.integrate as integrate
import scipy.optimize as optimize
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import axes3d
from numpy import cos, cosh, exp, pi, sin, sinh
from scipy import integrate, optimize


# Helpt dit?
def function_operation(h, f, g):
    return lambda z: h(f(z), g(z))


def scalar_multiplication(s, f):
    return lambda z: s * f(z)


def multiplication(f, g):
    return lambda z: math.prod([func(z) for func in [f, g]])


# Wavenumber: 2pi/wavelength lambda


# Wavelength: 33.8
wave_length = 33.8

sea_model = {
    "mild": (5.7, 1.5, 33.8),
    "medium": (8.6, 4.1, 76.5),
    "rough": (11.4, 8.5, 136),
}

wave_period, wave_amplitude, wave_length = sea_model["mild"]

# (Radian) frequency: ?
sigma = (2 * np.pi) / wave_period

#
T = 20

# I
# I = 490.7385
R_Outer_Shell = 5 / 2
R_Inner_Shell = 3 / 2
D_Shell = 0.15
I_Iron = pi / 4 * (R_Outer_Shell**4 - (R_Outer_Shell - D_Shell) ** 4) + pi / 4 * (
    (R_Inner_Shell + D_Shell) ** 4 - (R_Inner_Shell) ** 4
)
I_Concrete = pi / 4 * ((R_Outer_Shell - D_Shell) ** 4 - (R_Inner_Shell + D_Shell) ** 4)

I = I_Iron + I_Concrete
# I = 500
# E
# E = 210 * 10**9
# EI
# EI = E * I
EI = (I_Iron) * 210 * 10**9 + I_Concrete * 20 * 10**9

# To calculate the density
A_Steel = pi * (R_Outer_Shell**2 - (R_Outer_Shell - D_Shell) ** 2) + pi * (
    (R_Inner_Shell + D_Shell) ** 2 - (R_Inner_Shell) ** 2
)
A_Concrete = pi * ((R_Outer_Shell - D_Shell) ** 2 - (R_Inner_Shell + D_Shell) ** 2)
A_Air = pi * (R_Inner_Shell) ** 2
A_Total = pi * (R_Outer_Shell) ** 2

rho_steel = 7850
rho_concrete = 2300
rho_air = 1.225

rho_material = (
    A_Steel * rho_steel + A_Concrete * rho_concrete + A_Air * rho_air
) / A_Total


# Cilinder surface intersection
A = pi * (R_Outer_Shell) ** 2 - pi * (R_Inner_Shell) ** 2


kappa = 2 * np.pi / wave_length


# Water Density: 1025 kg/m^3
rho_water = 1025

# Inertia coefficient: 1+C_a ~ 8.66
# C_m = 8.66
C_m = 1.5
# Drag coefficient: 1.17
C_D = 1.17

# Cylinder diameter: ?
D = 5


# Horizontal bottom: < 50
H = 50

# l Length of the windmill 150
l = 150

# density of steel and water
rho_water = 1030


# Function that calculates phi
# @cache
# def calc_phi(z, x, t):
#     phi = (
#         sigma
#         / kappa
#         * a
#         * np.cosh(kappa * (z + H))
#         / np.sinh(kappa * H)
#         * np.sin((kappa * x) + (sigma * t))
#     )
#     return phi


# # Function that calculates the differential of phi
# @cache
# def diff_phi(z, x, t):
#     d_phi = (
#         sigma
#         * kappa
#         * a
#         * np.cosh(kappa * (z + H))
#         / np.sinh(kappa * H)
#         * np.cos((kappa * x) + (sigma * t))
#     )
#     return d_phi


# # Wave force function/morrison equation. Use this one to calculate wave forcing
# @cache
# def wave_force(z, rho, t):
#     F = (np.pi / 4 * rho_water * C_m * D**2 * diff_u(z, t)) + (
#         1 / 2
#     ) * rho_water * C_D * D * u(z, t) * np.abs(u(z, t))
#     return F


# # wave force function
# @cache
# def wave_force_var_dens(z, rho, t):
#     F = [[] for i in range(len(rho))]
#     for j in range(len(rho)):
#         F[j] = (np.pi / 4 * rho[j] * C_m * D**2 * diff_u(z, t)) + (1 / 2) * rho[
#             j
#         ] * C_D * D * u(z, t) * np.abs(u(z, t))
#     return F


# ! Niet handig
# # Function that calculates u. This is simply a function that is used in the wave forcing function
# def u(z, t):
#     u = sigma * a * np.cosh(kappa * (z + H)) / np.sinh(kappa * H) * np.cos(sigma * t)

#     # ! Error checker
#     # print(f"The value for u is {u} \n ")
#     return u


# function for the differential of u in z. This is simply a function that is used in the wave forcing function
# def diff_u(z, t):
#     diff_u = (
#         -(sigma**2)
#         * a
#         * np.cosh(kappa * (z + H))
#         / np.sinh(kappa * H)
#         * np.sin(sigma * t)
#     )
#     return diff_u


# # find force at wave position z at time t.
# force = np.empty((len(t), len(z)))
# for k in range(len(t)):
#     for i in range(len(z)):
#         force[k][i] = wave_force(i, rho_water, k)

# end = 1500
# step2 = 1
# rho = np.arange(1000, end + 1, step2)

# force_var_dens = np.empty((len(t), len(z), len(rho)))
# for k in range(len(t)):
#     for i in range(len(z)):
#         for y in range(len(rho)):
#             force_var_dens[k][i] = wave_force(i, rho[y], k)


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


# def test_q(x):
#     return np.sin(x)


# # lambda's for the beam equation
# # zeros = find_roots(f, 100, 0, 10)
# # print(zeros)


# def test_func(x):
#     return x**5


# # Application of simpson's rule to find the integral of a function.
# def simps_rule(func, a, b, n=100):
#     h = (b - a) / n
#     step = a
#     res = 0
#     for i in range(n):
#         step = a + i * h
#         res += np.abs(func(step) + 4 * func((step + step + h) / 2) + func(step + h))
#     res = (h / 6) * res
#     return res


# @cache
# def simps_rule_lambda_n(func, a, b, lambda_n, n=100, *args):
#     h = (b - a) / n
#     step = a
#     res = 0
#     for i in range(n):
#         step = a + i * h
#         res += np.abs(
#             func(step, lambda_n, *args)
#             + 4 * func((step + step + h) / 2, lambda_n, *args)
#             + func(step + h, lambda_n, *args)
#         )
#     #        print("res simps_rule_lambda_n = ", res)
#     res = (h / 6) * res
#     return res


# @cache
# def simps_rule_lambda_n_t(func, a, b, lambda_n, t, n=100, *args):
#     #    print("b = ",b, " a = ", a, " lambda_n = ", lambda_n, " t = ", t, " n = ", n)
#     h = (b - a) / n
#     #    print(h)
#     step = a
#     res = 0
#     for i in range(n):
#         step = a + i * h
#         res += np.abs(
#             func(step, t, lambda_n, *args)
#             + 4 * func((step + step + h) / 2, t, lambda_n, *args)
#             + func(step + h, t, lambda_n, *args)
#         )
#     #        print("res simps_rule_lambda_n_t = ", res)
#     res = (h / 6) * res
#     # print("res = ", res)
#     return res


# # print(Z_n(1))
# @cache
# def Z_n_sq(x: float, lambda_n: float = 1.0, C_n: float = 1.0, *args):
#     term1 = np.cos(lambda_n * x) - np.cosh(lambda_n * x)
#     term2 = (np.cos(lambda_n * l) - np.cosh(lambda_n * l)) / (
#         np.sin(lambda_n * l) - np.sinh(lambda_n * l)
#     )
#     term3 = np.sin(lambda_n * x) - np.sinh(lambda_n * x)
#     return (C_n * (term1 - term2 * term3)) ** 2


@cache
def frequency_equation(x: float, l: float = 150.0) -> float:
    return (np.exp(x * l) + np.exp(-x * l)) / 2 * np.cos(x * l) + 1


@cache
def find_lambdas(func, n):

    counter = 0
    zeros = []
    stepsize = (0.3043077 * 1.1) / 10000
    A = 0
    B = stepsize
    while counter < n:
        A += stepsize
        B += stepsize
        if func(A) * func(B) < 0:
            zeros.append(optimize.bisect(func, A, B))
            counter += 1
    return zeros


##print(len(find_lambdas(frequency_equation, 1000, 0, 2)))


# @cache
# def func1(x, t, lambda_n, rho=rho[0], *args):
#     return wave_force(x, rho, t) * Z_n(x, lambda_n)


# @cache
# def Q_n(t: float, lambda_n, *args):
#     return simps_rule_lambda_n_t(func1, 0, l, lambda_n, t)


# @cache
# def func2(tau, t, lambda_n: float, *args):
#     #    print("func2 left part: Q_n(tau, lambda_n) = ", Q_n(tau, lambda_n))
#     #    print("func2 right part: ", np.sin(lambda_n*(t-tau)))
#     return Q_n(tau, lambda_n) * np.sin(lambda_n * (t - tau))


# @cache
# def func_b(Z_n, lambda_n, *args):
#     return simps_rule_lambda_n(Z_n_sq, a=0, b=H, lambda_n=lambda_n)


# @cache
# def small_q(t_step, lambda_n, *args):
#     q_n = (
#         1 / (rho_material * A * func_b(Z_n, lambda_n) * lambda_n)
#     ) * simps_rule_lambda_n_t(func2, a=0, b=t_step, lambda_n=lambda_n, t=t_step)
#     #    print("left part small_q", (1/(rho_material*A*func_b(Z_n,lambda_n)*lambda_n)))
#     #    print("right part small_q", simps_rule_lambda_n_t(func2, a=0, b=t_step, lambda_n=lambda_n, t=t_step))
#     return q_n

# ! not in use anymore

# @cache
# def w(x, t, lambda_list: list, *args):
#     w = 0
#     n = len(lambda_list)
#     count = 1
#     for lambda_n in lambda_list:
#         w += Z_n(x=x, lambda_n=lambda_n) * small_q(t_step=t, lambda_n=lambda_n)
#         count += 1
#     #    print("w =", w)
#     return w


# * funtion that calculates the beta used in the document for some lambda
def beta_lambda_n(lambda_n) -> float:
    return (rho_material * A * lambda_n**4) / (EI)


# * The function to caculate the space dependent part of the SOV
@cache
def Z_n(lambda_n: float = 1.0, *args):
    wn = lambda_n * l
    shinh_wn = sinh(wn)
    cosh_wn = cosh(wn)
    C_n = (cosh_wn + cos(wn)) / (shinh_wn + sin(wn))
    res = (
        lambda z: ((1 - C_n) * exp(lambda_n * z) + (1 + C_n) * exp(-lambda_n * z)) / 2.0
        + C_n * sin(lambda_n * z)
        - cos(lambda_n * z)
    )

    # ! Error Checker
    # print(
    #     f"The value for lambda_n is \n {lambda_n} \n which provides a value fo C_n of:{C_n}\n"
    # )
    return res


def integral(func, a, b):
    return integrate.quad(func, a=a, b=b, limit=100)[0]


@cache
def Q_n(lambda_n, rho_water=rho_water, A=A, T=T):

    # A = (pi / 4) * rho_water * C_m * D**2 * sigma**2 * a
    # * introduced constants:
    k = kappa

    # # * u is the function for the wave forcing. Here initialized as lambda function
    # u_drag = lambda z: (
    #     (np.exp(kappa * (z + H)) + np.exp(-kappa * (z + H))) / 2 if z <= 50 else 0
    # )

    # * Z_eq is the function for this lambda
    Z_eq = Z_n(lambda_n=lambda_n)

    # * ## Inertia Calculations ## * #

    # * Inertia part for Q
    # * Integrate the u_inertia. It remains constant for t.
    D_inertia_const = integral(
        lambda z: (exp(k * (z)) + exp(-k * (z))) / 2 * Z_eq(z), 0, H
    )
    C_inert = (
        -D_inertia_const * A * rho_water * C_m * sigma**2 * wave_amplitude / sinh(k * H)
    )

    # ! Error Checker
    # print(f"The C_inert is {C_inert}\n")

    # * The inertia itself is dependent on t. So it is put in a lambda function
    D_inertia = lambda t: C_inert * sin(sigma * t)

    # * ## Drag Calculations ## * #
    D_Drag_const = integral(
        lambda z: ((1 / 4) * exp(k * (z)) + exp(-k * (z))) ** 2 * Z_eq(z),
        a=0,
        b=H,
    )
    C_drag = (
        D_Drag_const
        * (1 / 2)
        * rho_water
        * C_D
        * D
        * sigma
        * wave_amplitude
        / (sinh(k * H)) ** 2
    )

    # ! Error Check
    # print(f"The values for D_Drag_const is {D_Drag_const}\n")

    # * Integrate the u_drag. It remains constant for t
    D_Drag = lambda t: C_drag * cos(sigma * t) * abs(cos(sigma * t))

    # * Return the sum of both functions
    Q = function_operation(add, D_inertia, D_Drag)

    return Q


# def a_n_inef(a_n_const,t):

#     u = lambda z,t: sigma * a * ((exp(k*(z+H)) + exp(-k*(z+H))) / (exp(k*H) - exp(-k*H))) * cos(sigma*t)
#     u_diff = - sigma**2 * a * ((exp(k*(z+H)) + exp(-k*(z+H))) / (exp(k*H) - exp(-k*H))) * sin(sigma*t)
#     np.pi / 4 * rho_water * C_m * D**2 * u(z, t) + (1 / 2) * rho_water * C_D * D * u(z, t) * np.abs(u(z, t))

#     return integral(lambda z: )


# ? Function needs to be created top level in order to not be overwritten
@cache
def a_n(mu_n, a_const, n, Q_n):
    # * Return time dependent part of the sov
    # ! Error Check
    # val = integrate.quad(lambda tau: Q_n(tau) * np.sin(mu_n * tau), a=0, b=1, limit=300)
    # print(f"hey this is the value for this A_N's constant cos part at one second {val}")
    return lambda t, n=n: a_const * (
        np.sin(mu_n * t)
        * integrate.quad(
            lambda tau: Q_n(tau) * np.cos(mu_n * tau), a=0, b=t, limit=300
        )[0]
        - np.cos(mu_n * t)
        * integrate.quad(
            lambda tau: Q_n(tau) * np.sin(mu_n * tau), a=0, b=t, limit=300
        )[0]
    )


def b_list(lambda_list: list):
    b = []

    for lambda_n in lambda_list:
        Zn = Z_n(lambda_n=lambda_n)
        Zn_squared = multiplication(Zn, Zn)

        b.append(integral(Zn_squared, a=0, b=l))

        # ! Error Checker

    return b


def BEQ(
    t_end: float = 5,
    dt: float = 1,
    l: int = 150,
    dl: float = 15,
    A=A,
    rho_water=rho_water,
    T=T,
) -> list:
    # * Constants
    # A = np.pi * (D / 2) ** 2

    # # ! Error Checker
    # print(f"Creating a list of eigenvalues")
    lambda_list = find_lambdas(frequency_equation, 5)

    # ! Error Checker
    # print(f"amount of lambdas is {len(lambda_list)}")
    # print(f"\n The lambdas are: \n {lambda_list}")

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

    # # ! Error Checker
    # print(f"Creating a list for Z_n and a_n")
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
        mu_n = (lambda_n) ** 2 * np.sqrt(EI / (rho_material * A))
        # A_an = np.pi * rho_water * C_m * D**2
        # B_an = (1 / 2) * rho_water * C_D * D
        a_const = 1 / (rho_material * A * mu_n * b[count])

        # * a_n_list constains lambda functions of a_n(t). Input for the functions is t
        a_n_list.append(
            a_n(
                a_const=a_const,
                mu_n=mu_n,
                n=count,
                Q_n=Q_n(lambda_n=lambda_n, rho_water=rho_water),
            )
        )

    # ! Error checkers
    # print(f"\nthe first value for Z_n at the top is \n {(Z_n_list[0])(150)}\n")
    # print(f"the first a_n is {print(a_n[0])}")
    # print(f"\nthe list of lambda's is \n {lambda_list}\n")

    # * structure of output for the function BEQ
    # [ [time1,[ [x1,z1], [x2,z2],...  ]],[time2,[ [x1,z1], [x2,z2],...  ]],...  ]

    for t_step in time_steps:

        # ! Error checker
        # print(f"t_step ={t_step}")

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
            Z_lambda = Z_n_list[count]

            # ! Error checker a_lambda and z_lambda
            # print(f"a_lambda is {a_lambda} ")
            # print(f"z_lambda(150) is {Z_lambda(150)} ")

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
for time in bop:
    print(f"at time {time[0]} the beam has deviations{time[1]}")


# * The maximum deviation
@cache
def max_dev(rho_w=rho_water, t_end=10):
    dt = 1
    l = 150
    dl = 10

    U_t = BEQ(t_end=t_end, dt=dt, l=l, dl=dl, rho_water=rho_w, T=T)
    bigollist = []
    max_val = 0

    for bop in U_t:
        bigollist += bop[1]

    bigollist.sort()

    max_val = (bigollist)[-1]
    min_val = (bigollist)[0]
    return max(abs(min_val), abs(max_val))


dt = 1 / 10
t_end = 5
l = 150
dl = 10


def xtu_diagram(t_end=t_end, dt=dt, l=l, dl=dl):
    times = np.arange()


def Rho_Water(dT, dS, rws):
    return rws * (2 - dT + dS) / 2


def rho_umax_diagram():
    dT_list = np.linspace(0, 0.05, 10)
    dS_list = np.linspace(0, 0.05, 10)


# *
@cache
def Vary_Rho():
    dt = 1 / 10
    t_end = 5
    l = 150
    dl = 10
    rho_water_start = 1030
    max_list = np.empty((10, 10))
    dT_list = np.linspace(0, 0.05, 10)
    dS_list = np.linspace(0, 0.05, 10)

    for dt_count, dT in enumerate(dT_list):
        temp_list = np.empty(10)
        for ds_count, dS in enumerate(dS_list):
            rho_w = Rho_Water(dT=dT, dS=dS, rws=rho_water_start)
            temp_list[ds_count] = max_dev(rho_w=rho_w)
        max_list[dt_count] = temp_list
    return max_list


# maximoem = max_dev(t_end=15)
# print(f"\n THE MAX Deviations Is: {maximoem} \n")


# varlis = Vary_Rho()
# print(f"the list of maxes with varied rhos are: {varlis}")


# for xyz in varlis:
#     X, Y, Z = xyz
def DT_DS_UMAX_Diagram():
    plt.style.use("_mpl-gallery")

    # Make data.
    X = np.linspace(0, 0.05, 10)
    Y = np.linspace(0, 0.05, 10)
    Z = Vary_Rho()

    x, y = np.meshgrid(X, Y)

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, Z, cmap="cool")

    ax.set_xlabel("DT")
    ax.set_ylabel("DS")
    ax.set_zlabel("u_max")

    # Plot the surface.
    plt.show()


DT_DS_UMAX_Diagram()
# z = BEQ()
# filename = "results.csv"
# with open(filename, 'w') as csvfile:
#     # creating a csv dict writer object
#     writer = csv.DictWriter(csvfile)

#     # writing data rows
#     writer.writerows(z)


# test, answer should be 12
# print(simps_rule(test_func, 0, 2, 1))
print("wtf")

# print(find_lambdas(frequency_equation, 10, 0, .1))


@cache
def rho_water_calc(T):
    a = -2.8054253 * 10**-10
    b = 1.0556302 * 10**-7
    c = -4.6170461 * 10**-5
    d = -0.0079870401
    e = 16.945176
    f = 999.83952
    g = 0.01687985
    # R   ange of validity : [-30 ; 150] c
    rho_water = (((((a * T + b) * T + c) * T + d) * T + e) * T + f) / (1 + g * T)
    return rho_water


@cache
def Rho_Umax(salinity):
    X = np.linspace(0, 40, 40)
    Y = np.zeros(len(X))
    for count, temp in enumerate(X):

        # ! Error Checker
        # bro = rho_water_calc(temp)
        # print(f"the value for rho_water is now {bro}")

        Y[count] = max_dev(rho_w=(rho_water_calc(temp) + salinity))

    # print(f"The Y array looks like: \n {Y}")
    return X, Y


@cache
def Operation_Salt():
    # Make data.

    fig, ax = plt.subplots(figsize=(6, 6))

    salts = np.linspace(0, 100, 5)
    for salt in salts:
        X, Y = Rho_Umax(salt)
        ax.plot(X, Y, label=f"Extra salinity of {salt}")
    ax.set_xlabel("Temperature in C")
    ax.set_ylabel("Max deviation in m")
    ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncols=2,
        mode="expand",
        borderaxespad=0.0,
    )
    plt.show()


# Operation_Salt()


def get_height(beq, count, time):
    loc = [
        (index1, index2)
        for index1, value1 in enumerate(beq)
        for index2, value2 in enumerate(value1)
        if value2 == time
    ]
    return beq[loc[0[0]]][1][count]


def colormap():

    plt.style.use("_mpl-gallery")

    # Make data.
    T = np.linspace(0, t_end, t_end / dt)
    X = np.linspace(0, l + dl, int((l) / dl))
    X_counter = np.linspace(0, len(X), 1)
    x, y = np.meshgrid(X, T)
    U = np.zeros(len(T))

    vals = np.array(BEQ())

    U = get_height(vals, X_counter, T)

    print(U)

    # fig = plt.figure(figsize=(10, 8))

    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(x, y, Z, cmap="cool")

    # ax.set_xlabel("DT")
    # ax.set_ylabel("DS")
    # ax.set_zlabel("u_max")
    # print(vals)

    # fig, ax = plt.subplots()
    # im = ax.imshow(data2d)
    # ax.set_title(
    #     "Pan on the colorbar to shift the color mapping\n"
    #     "Zoom on the colorbar to scale the color mapping"
    # )

    # fig.colorbar(im, ax=ax, label="Interactive colorbar")

    # plt.show()


# colormap()
