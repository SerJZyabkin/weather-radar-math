#!/usr/bin/env python3
"""
`scattering_simulation.calculation.dielectric_permittivity.particle_simulation.rain.physical_parameters.py` содержит
набор методов расчета различных моментов для функции распреления числа частиц по размерам.
Расчет повторяет методику, представленную в главе 2.1.1 из учебника Weather Radar Polarimetry авторства Guifu Zhang.
"""

import numpy as np

def total_number_consentration(drop_size_distribution, min_drop_size: float, max_drop_size: float,
                               delta_drop_size: float) -> int:
    """
    Метод интегрирования распределения размеров частиц дождя DSD для нахождения числа полной концентрации частиц дождя
    для заданного распределения. Пределы и шаг интегрирования можно изменять. DSD можно получить в файле .drop_size.py.

    :param drop_size_distribution: N(D) - функциональная зависимость распеределения размеров капель (DSD)
    :param min_drop_size: Dmin - нижний предел интегрирования по размерам капель, мм
    :param max_drop_size: Dmax - верхний предел интегирования по размерам капель, мм
    :param delta_drop_size: dD - шаг интегрирования по размерам капель
    :return: Nt - полная концентрация, # m^(-3) (Математический символ # обозначает число частиц в множестве)
    """
    diameter_arr = np.arange(min_drop_size, max_drop_size, delta_drop_size)
    return np.ceil(np.sum(drop_size_distribution(diameter_arr) * delta_drop_size))



def rain_water_content(drop_size_distribution, min_drop_size: float, max_drop_size: float,
                               delta_drop_size: float) -> int:
    """
    Метод интегрирования распределения размеров частиц дождя DSD для нахождения водности дождя W для заданного
    распределения. Пределы и шаг интегрирования можно изменять. DSD можно получить в файле .drop_size.py.

    :param drop_size_distribution: N(D) - функциональная зависимость распеределения размеров капель (DSD)
    :param min_drop_size: Dmin - нижний предел интегрирования по размерам капель, мм
    :param max_drop_size: Dmax - верхний предел интегирования по размерам капель, мм
    :param delta_drop_size: dD - шаг интегрирования по размерам капель
    :return: Nt - полная концентрация, # m^(-3) (Математический символ # обозначает число частиц в множестве)
    """
    diameter_arr = np.arange(min_drop_size, max_drop_size, delta_drop_size)
    return np.pi / 6.e3 * np.ceil(np.sum(np.power(diameter_arr, 3) * drop_size_distribution(diameter_arr)
                                         * delta_drop_size))



