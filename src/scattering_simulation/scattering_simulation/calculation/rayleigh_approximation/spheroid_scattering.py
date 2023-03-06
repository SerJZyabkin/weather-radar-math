#!/usr/bin/env python3
"""
`scattering_simulation.calculation.rayleigh_approximation.spheroid_scattering.py` содержит методы для
"""

from numpy import pi, power, complex, sqrt, arctan, log, exp
from scipy.integrate import quad
from ..particle_properties.rain.drop_shape import get_shape


def shape_factor_z(a, b):
    gamma = b / a
    if gamma < 1.:  # a > b - сплюснутый сферойд
        sqr_g = 1. / power(gamma, 2.) - 1.
        g = sqrt(sqr_g)
        return (1. + sqr_g) / sqr_g * (1. - arctan(g) / g)

    # Иначе вытяюнутый сферойд
    sqr_e = 1. - 1. / power(gamma, 2.)
    e = sqrt(sqr_e)
    return (1. - sqr_e) / sqr_e * (-1. + log((1 + e) / (1 - e)) / (2 * e))


def shape_factor_xy(a, b):
    return 0.5 * (1 - shape_factor_z(a, b))


def backscattering_cross_sections(a: float, b: float, wavelength: float, dielectric_constant_relative: complex):
    """
    Расчет поперечного сечения обраного рассеяния для релеевской аппроксимации рассеяния от одной частицы. Формула
    3.65 взата из учебника Weather Radar Polarimetry авторства Guifu Zhang.

    :param wavelength: lambda - длина волны
    :param dielectric_constant_relative: - epsilon_r - относительная диэлектрическая постоянная вещества частицы
    :return: sigma_b - поперечное сечение обратного рассеяния
    """

    k = 2. * pi / wavelength
    multi = pi * power(k, 4.) * power(a, 4.) * power(b, 2)
    return [multi * power((dielectric_constant_relative - 1.) / 3 / (1 + (dielectric_constant_relative - 1) * L), 2.)
            for L in [shape_factor_xy(a, b), shape_factor_z(a, b)]]


def backscattering_amplitude(a: float, b: float, wavelength: float, dielectric_constant_relative: complex):
    k = 2. * pi / wavelength
    multi = power(k, 2.) * power(a, 2.) * b
    s_a = multi * (dielectric_constant_relative - 1.) / 3 / (1 + (dielectric_constant_relative - 1) *
                                                                  shape_factor_xy(a, b))
    s_b = multi * (dielectric_constant_relative - 1.) / 3 / (1 + (dielectric_constant_relative - 1) *
                                                                  shape_factor_z(a, b))
    return s_a, s_b
