#!/usr/bin/env python3
"""
Скрипт для проверки работы функция расчета диэлектирческой постоянной и индекса рефракции для льда при помоши
повторения графика, аналогичного графику 2.22 из учебника Weather Radar Polarimetry авторства Guifu Zhang
"""

import numpy as np
from scattering_simulation.calculation.dielectric_permittivity.lossy_medium import get_ice_dielectric_constant, \
    get_air_dielectric_constant
from scattering_simulation.calculation.dielectric_permittivity.medium_mixture import get_mixing_maxwell_garnett, \
    get_mixing_ponder_vansandern
import matplotlib.pyplot as plt


wave_length = 0.032
temperature = -10

eps_ice = get_ice_dielectric_constant(wave_length, temperature)
eps_air = get_air_dielectric_constant(wave_length)
fractions_air = np.arange(0, 1, 0.001)
fractions_ice = 1 - fractions_air

vals = get_mixing_maxwell_garnett(eps_air, eps_ice, fractions_air, fractions_ice)
plt.plot(fractions_ice, vals.real, 'k-.')

vals = get_mixing_maxwell_garnett(eps_ice, eps_air, fractions_ice, fractions_air)
plt.plot(fractions_ice, vals.real, 'k--')

vals = np.array([get_mixing_ponder_vansandern(eps_air, eps_ice, f1, f2) for f1, f2 in zip(fractions_air, fractions_ice)])
plt.plot(fractions_ice, vals.real, 'k')

plt.show()