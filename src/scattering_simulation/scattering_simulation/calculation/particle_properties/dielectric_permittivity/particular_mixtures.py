#!/usr/bin/env python3
"""
`scattering_simulation.calculation.dielectric_permittivity.particular_mixtures`
"""
import numpy as np
from .lossy_medium import get_ice_dielectric_constant, get_water_dielectric_constant, get_air_dielectric_constant
from .medium_mixture import get_mixing_three_components


def get_mixing_for_wet_snow(wave_length: float, temperature: float, ro_ds: float, gamma_w: float,
                            d_ds: float, d_w: float) -> complex:
    # Расчет плотности воды после таяния при предположении, что масса частицы сохраняется, а потом и тающего снега
    _ro_w = ro_ds * np.power(d_ds, 3.) / np.power(d_w, 3.)
    _ro_ws = ro_ds * (1 - np.square(gamma_w)) + _ro_w * np.square(gamma_w)

    # теперь, когда известны плотности для всех трех компонент, можно получить их доли объемов f
    _f_w = gamma_w * _ro_ws / _ro_w
    _f_i = (1 - gamma_w) * _ro_ws / ro_ds
    _f_a = 1 - _f_w - _f_i

    # Диэлектрические проницаемости компонентов
    _eps_w = get_water_dielectric_constant(wave_length, temperature)
    _eps_i = get_ice_dielectric_constant(wave_length, temperature)
    _eps_a = get_air_dielectric_constant(wave_length)

    # Расчет итоговой диэлектрической проницаемости по полученным долям компонент
    return get_mixing_three_components(_eps_w, _eps_i, _eps_a, _f_w, _f_i, _f_a)
