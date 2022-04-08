#!/usr/bin/env python3
"""
`scattering_simulation.calculation.dielectric_permittivity.lossy_medium` содержит набор функций для расчета
диэлектической проницаемости для однокомпонентной среды с потерями. В интересах данной работы рассматриваются лишь
среды, состоящие из жидкой воды или льда. Расчет повторяет методику, представленную в главе 2.2.2 из учебника Weather
Radar Polarimetry авторства Guifu Zhang.
"""
import numpy as np
from src.scattering_simulation.scattering_simulation.constants import LIGHTSPEED, EPSILON_ZERO
from src.scattering_simulation.scattering_simulation.calculation.common_conversions import convert_to_frequency


def get_air_dielectric_constant(wavelength: float) -> complex:
    _re_epsilon = 1.0006
    _sigma = 1e-12
    _omega = 2 * np.pi * convert_to_frequency(wavelength)
    return _re_epsilon - 1j * _sigma / _omega


def get_water_dielectric_constant(wavelength: float, temperature: float) -> complex:
    """
    Формула расчета комплексной диэлектрической постоянной для текущих значений длины волны и температуры. Расчет
    происходит на основе релаксационной модели Дюбуа (Debya relaxation model), модефицированной Коле в 1941, и по
    коэффициентам для воды, полученным Реем в 1973.

    :param wavelength: длина волны в метрах
    :param temperature: температура воды в градусах Цельсия
    :return: комплексная диэлектрическая постоянная воды для текущих значений длины волны и температуры
    """
    _epsilon_s = 78.54 * (1.0 - 4.579e-3 * (temperature - 25) + 1.19e-5 * np.power(temperature - 25, 2) -
                          2.8e-8 * np.power(temperature - 25, 3))
    _epsilon_inf = 5.27137 + 2.16474e-2 * temperature - 1.31198e-3 * np.square(temperature)
    _alpha = -16.8129 / (temperature + 273) + 6.09265e-2
    _lambda_s = 3.3836e-6 * np.exp(2513.98 / (temperature + 273))
    _sigma = 1.1117e-4

    # Расчет диэлектрической постоянной по формуле
    return __get_dielectic_constant__(wavelength, _epsilon_s, _epsilon_inf, _alpha, _lambda_s, _sigma)


def get_water_reflection_index(wavelength: float, temperature: float) -> complex:
    """
    Процедура расчета комплексного индекса рефракции для воды по значению диэлектрической постоянной из метода
    get_ice_dielectric_constant

    :param wavelength: длина волны в метрах
    :param temperature: температура воды в градусах Цельсия
    :return: комплексный индекс рефракции для воды для текущих значений длины волны и температуры
    """
    return np.sqrt(get_water_dielectric_constant(wavelength, temperature))


def get_ice_dielectric_constant(wavelength: float, temperature: float) -> complex:
    """
    Формула расчета комплексной диэлектрической постоянной для текущих значений длины волны и температуры. Расчет
    происходит на основе релаксационной модели Дюбуа (Debya relaxation model), модефицированной Коле в 1941, и по
    коэффициентам для льда, полученным Реем в 1973.

    :param wavelength: длина волны в метрах
    :param temperature: температура льда в градусах Цельсия
    :return: комплексная диэлектрическая постоянная льда для текущих значений длины волны и температуры
    """
    # Табличные значения для льда в соответствии с работой Ray 1972
    _epsilon_s = 203.168 + 2.5 * temperature + 0.15 * np.square(temperature)
    _epsilon_inf = 3.168
    _alpha = 0.288 + 5.2e-3 * temperature + 2.3e-4 * np.square(temperature)
    _lambda_s = 9.990288e-6 * np.exp(6643.5 / (temperature + 273))
    _sigma = 1.1156e-13

    # Расчет диэлектрической постоянной по формуле
    return __get_dielectic_constant__(wavelength, _epsilon_s, _epsilon_inf, _alpha, _lambda_s, _sigma)


def get_ice_reflection_index(wavelength: float, temperature: float) -> complex:
    """
    Процедура расчета комплексного индекса рефракции для льда по значению диэлектрической постоянной из метода
    get_ice_dielectric_constant

    :param wavelength: длина волны в метрах
    :param temperature: температура льда в градусах Цельсия
    :return: комплексный индекс рефракции для льда для текущих значений длины волны и температуры
    """
    return np.sqrt(get_ice_dielectric_constant(wavelength, temperature))


def __get_dielectic_constant__(wavelength, epsilon_s, epsilon_inf, alpha, lambda_s, sigma) -> complex:
    """
    Вспомогательная формула расчета комплексной диэлектрической постоянной для текущих входных значений для
    использования в методах get_ice_dielectric_constant и get_water_dielectric_constant.

    :param wavelength: длина волны в метрах
    :param epsilon_s:
    :param epsilon_inf:
    :param alpha:
    :param lambda_s:
    :param sigma:
    :return: комплексная диэлектрическая постоянная для выбранных входных значений
    """
    # Предварительный расчет промежуточных величин
    _delta_epsilons = epsilon_s - epsilon_inf
    _lambda_ratio = lambda_s / wavelength
    _lambda_pow_1 = np.power(_lambda_ratio, 1 - alpha)
    _lambda_pow_2 = np.power(_lambda_ratio, 2 * (1 - alpha))
    _angle = alpha * np.pi / 2

    # Расчет комплексной диэлектрической проницаемости
    _re_epsilon = (1 + _lambda_pow_1 * np.sin(_angle)) * _delta_epsilons / (1 + 2 * _lambda_pow_1 * np.sin(_angle) +
                                                                            _lambda_pow_2) + epsilon_inf
    _im_epsilon = sigma * wavelength / (2 * np.pi * LIGHTSPEED * EPSILON_ZERO)
    _im_epsilon += _delta_epsilons * (_lambda_pow_1 * np.cos(_angle)) / (1 + 2 * _lambda_pow_1 * np.sin(_angle) +
                                                                         _lambda_pow_2)
    # Возвращаем полученное комплексное значение
    return _re_epsilon + 1j * _im_epsilon
