# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:52:55 2024

@author: Maris
"""
import numpy as np
# (radian) frequency: ?
sigma = 1

# wavenumber: ?
kappa = 1

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
