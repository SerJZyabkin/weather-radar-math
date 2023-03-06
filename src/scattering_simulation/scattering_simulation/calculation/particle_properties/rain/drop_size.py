#!/usr/bin/env python3
"""
`scattering_simulation.calculation.dielectric_permittivity.particle_simulation.rain.drop_size.py` содержит набор методов
для создания фабричных функций получения числа частиц в единице фиксированного объема в зависмости от размера капель.

Объявляются следующие распредления числа частиц по размерам:
    - Двухпараметрическое экспоненциальное распределение Маршалла-Палмера
    - Трехпараметрическое гамма-распределение Улбрича [Ulbrich, C. W., 1983. Natural variations in the analytical form
        of the raindrop size distribution. Journal of Climate and Applied Meteorology, 22, 1764–1775.]

Расчет повторяет методику, представленную в главе 2.1.1 из учебника Weather Radar Polarimetry авторства Guifu Zhang.
"""

import numpy as np
from math import gamma

def factory_exponential_distribution(intersept: float, slope: float):

    def exponential_distribution(particle_size: float):
        return intersept * np.exp(-slope * particle_size)

    return exponential_distribution


def factory_gamma_distribution(number_concentration: float, shape: float, slope: float):
    """

    :param number_concentration: N0 - параметр распределения, равный мат. ожиданию случайной величины при нулевом
        входном значении (размере частиц) и нулевом shape параметре, размерность m^(−3) mm^(shape − 1)
    :param shape: mu - параметр формы гамма распределения, безразмерная величина
    :param slope: Lambda - параметр распределения, множитель экспоненты, размерность mm^(−1)
    :return: Функцианальную зависимость числа частиц N в зависимости от их размера D по гамма-распределению
    """

    def gamma_distribution(particle_size: float):
        return number_concentration * np.power(particle_size, shape) * np.exp(-slope * particle_size)

    return gamma_distribution

def factory_normalized_gamma_distribution(number_concentration: float, shape: float, normalized_particle_size: float):
    """
    Формула (7.62b) из Polarimetic Doppler Weather Radar

    :param number_concentration: - параметр распределения, равный мат. ожиданию случайной величины при нулевом
        входном значении (размере частиц) и нулевом shape параметре, размерность m^(−3) mm^(shape − 1)
    :param shape: mu - параметр формы гамма распределения, безразмерная величина
    :param normalized_particle_size: D0 - размерность mm
    :return: Функцианальную зависимость числа частиц N в зависимости от их размера D по нормализованному
        гамма-распределению
    """
    f_shape = 6 / np.power(3.67, 4) * np.power(3.67 + shape, shape + 4) / gamma(shape + 4)

    def normalized_gamma_distribution(particle_size: float):
        delta_particle_size = particle_size / normalized_particle_size
        return number_concentration * f_shape * np.power(delta_particle_size, shape) * np.exp(-(3.67 + shape) *
                                                                                              delta_particle_size)

    return normalized_gamma_distribution


def factory_normalized_exponential_distribution(intersept: float, slope: float, normalized_particle_size: float):

    def exponential_distribution(particle_size: float):
        delta_particle_size = particle_size / normalized_particle_size
        return intersept * np.exp(-slope * delta_particle_size)

    return exponential_distribution
