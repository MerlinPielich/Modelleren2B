# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:52:55 2024

@author: Maris
"""
import numpy as np
sigma = 1
kappa = 1
a = 1
rho = 1
C_m = 1
C_D = 1
D = 1
H = 1


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
