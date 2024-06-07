# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:52:55 2024

@author: Maris
"""
import numpy as np
from scipy import optimize

# (radian) frequency: ?
sigma = 1

# wavelength: ?
lamda = 1

# wavenumber: 2pi/wavelength lambda
kappa = 2 * np.pi / lamda

# wave amplitude: ?
a = 1

# water pressure: 1025 kg/m^2
rho = 1025

# inertia coefficient: 1+C_a ~ 8.66
C_m = 8.66

# drag coefficient: 1.17
C_D = 1.17

# cylinder diameter: ?
D = 5

# Horizontal bottom: < 50
H = 50


def calc_phi(z, x, t):
    phi = sigma / kappa * a * np.cosh(kappa * (z+H)) / \
        np.sinh(kappa*H) * np.sin((kappa * x)+(sigma * t))
    return phi


def diff_phi(z, x, t):
    d_phi = sigma * kappa * a * np.cosh(kappa * (z + H)) / \
        np.sinh(kappa * H) * np.cos((kappa * x) + (sigma * t))
    return d_phi


def wave_force(z, t):
    F = (np.pi / 4 * rho * C_m * D ** 2 * diff_u(z, t)) + \
        1/2 * rho * C_D * D * u(z, t) * np.abs(u(z, t))
    return F


def u(z, t):
    u = sigma * a * np.cosh(kappa * (z + H)) / \
        np.sinh(kappa * H) * np.cos(sigma * t)
    return u


def diff_u(z, t):
    diff_u = - sigma ** 2 * a * np.cosh(kappa * (z + H)) / \
        np.sinh(kappa*H) * np.sin(sigma * t)
    return diff_u


def f(x):
    return np.cosh(x)*np.cos(x)+1


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


def q(x):
    return np.sin(x)


print(find_roots(f, 100, 0, 100))


def func(x):
    return x ** 5


def simps_rule(func, a, b, n):
    h = (b-a)/n
    step = a
    res = 0
    for i in range(n):
        step = a + i * h
        res += func(step) + 4*func((step + step + h)/2) + func(step + h)
    res = (h/6) * res
    return res

# test, answer should be 12
# print(simps_rule(func, 0, 2, 1))
