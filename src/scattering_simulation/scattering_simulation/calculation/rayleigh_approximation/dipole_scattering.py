#!/usr/bin/env python3
"""
`scattering_simulation.calculation.rayleigh_approximation.dipole_scattering.py` содержит методы для
"""

from numpy import pi, power, complex


def total_cross_section(sphere_radius: float, wavelength: float, dielectric_constant_relative: complex):
    """
    Расчет полного поперечного сечения рассеяния для релеевской аппроксимации рассеяния от одной частицы. Формула
    3.64 взата из учебника Weather Radar Polarimetry авторства Guifu Zhang.

    :param sphere_radius: a - размер сферы, апприксимурующей исследуемую частицу
    :param wavelength: lambda - длина волны
    :param dielectric_constant_relative: - epsilon_r - относительная диэлектрическая постоянная вещества частицы
    :return: sigma_b - поперечное сечение обратного рассеяния
    """

    k = 2. * pi / wavelength
    return 8. * pi / 3. * power(k, 4.) * power(sphere_radius, 6.) * power((dielectric_constant_relative - 1.) /
                                                                          (dielectric_constant_relative + 2.), 2.)


def backscattering_cross_section(sphere_radius: float, wavelength: float, dielectric_constant_relative: complex):
    """
    Расчет поперечного сечения обраного рассеяния для релеевской аппроксимации рассеяния от одной частицы. Формула
    3.65 взата из учебника Weather Radar Polarimetry авторства Guifu Zhang.

    :param sphere_radius: a - размер сферы, апприксимурующей исследуемую частицу
    :param wavelength: lambda - длина волны
    :param dielectric_constant_relative: - epsilon_r - относительная диэлектрическая постоянная вещества частицы
    :return: sigma_b - поперечное сечение обратного рассеяния
    """

    k = 2. * pi / wavelength
    return 4. * pi * power(k, 4.) * power(sphere_radius, 6.) * power((dielectric_constant_relative - 1.) /
                                                                     (dielectric_constant_relative + 2.), 2.)


def absorbtion_cross_section(sphere_radius: float, wavelength: float, dielectric_constant_relative: complex):
    """
    Расчет поперечного сечения поглощения для релеевской аппроксимации рассеяния от одной частицы. Формула
    3.66 взата из учебника Weather Radar Polarimetry авторства Guifu Zhang.

    :param sphere_radius: a - размер сферы, апприксимурующей исследуемую частицу
    :param wavelength: lambda - длина волны
    :param dielectric_constant_relative: - epsilon_r - относительная диэлектрическая постоянная вещества частицы
    :return: sigma_b - поперечное сечение обратного рассеяния
    """

    k = 2. * pi / wavelength
    return (4. * pi / 3. * k * power(sphere_radius, 3.) * power(3. / (dielectric_constant_relative + 2.), 2.) *
            dielectric_constant_relative.imag)



