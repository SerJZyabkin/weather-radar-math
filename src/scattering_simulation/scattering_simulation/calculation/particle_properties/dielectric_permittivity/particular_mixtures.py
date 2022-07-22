#!/usr/bin/env python3
"""
`scattering_simulation.calculation.dielectric_permittivity.particular_mixtures`
"""
import numpy as np
from .lossy_medium import get_ice_dielectric_constant, get_water_dielectric_constant, get_air_dielectric_constant
from .medium_mixture import get_mixing_three_components
from ...common_constants import density_ice, density_water


def get_mixing_for_wet_snow(wave_length: float, temperature: float, density_ds: float, melt_percent: float) -> complex:
    """
    Расчет плотности воды после таяния при предположении, что масса частицы сохраняется, а потом и тающего снега

    :param wave_length:
    :param temperature:
    :param density_ds:
    :param melt_percent:
    :return:
    """
    _sqr_melt_percent = np.power(melt_percent, 2.)
    _density_ws = density_ds * (1 - _sqr_melt_percent) + density_water * _sqr_melt_percent

    # теперь, когда известны плотности для всех трех компонент, можно получить их доли объемов f
    _f_w = melt_percent * _density_ws / density_water
    _f_i = (1 - melt_percent) * _density_ws / density_ice
    _f_a = 1 - _f_w - _f_i

    # Диэлектрические проницаемости компонентов
    _eps_w = get_water_dielectric_constant(wave_length, temperature)
    _eps_i = get_ice_dielectric_constant(wave_length, temperature)
    _eps_a = get_air_dielectric_constant(wave_length)

    # Расчет итоговой диэлектрической проницаемости по полученным долям компонент
    return get_mixing_three_components(_eps_w, _eps_i, _eps_a, _f_w, _f_i, _f_a)
