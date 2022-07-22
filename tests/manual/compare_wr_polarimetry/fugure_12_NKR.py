#!/usr/bin/env python3
"""
Скрипт для проверки работы функция расчета диэлектирческой постоянной и индекса рефракции для льда при помоши
повторения графика, аналогичного графику 2.22 из учебника Weather Radar Polarimetry авторства Guifu Zhang
"""

import numpy as np
from scattering_simulation.calculation.particle_properties.dielectric_permittivity.lossy_medium import get_ice_dielectric_constant, \
    get_water_dielectric_constant
from scattering_simulation.calculation.particle_properties.dielectric_permittivity.medium_mixture import get_mixing_maxwell_garnett, \
    get_mixing_ponder_vansandern
import matplotlib.pyplot as plt


wave_length = 0.032
temperature = [-15, -10, -5, 0]
fractions_air = np.arange(0, 1, 0.001)
fractions_ice = 1 - fractions_air
fig, ((ax_re, ax_im)) = plt.subplots(1, 2)
lines_type = ['k', 'k--', 'k-.', 'k:']
leg = []
for step in range(4):
    eps_ice = get_ice_dielectric_constant(wave_length, temperature[step])
    eps_air = get_water_dielectric_constant(wave_length, temperature[step])

    vals = np.array([get_mixing_ponder_vansandern(eps_air, eps_ice, f1, f2) for f1, f2 in zip(fractions_air, fractions_ice)])
    ax_re.plot(fractions_air, vals.real, lines_type[step])
    ax_im.plot(fractions_air, vals.imag, lines_type[step])
    leg.append(f'T = {temperature[step]}°C')

# x_limits_arr = [[1e-3, 1], [1e-3, 1], [1e-3, 1], [1e-3, 1]]
# y_limits_arr = [[1, 1e2], [1, 2e1], [1, 1e2], [1e-1, 1e1]]
# y_labels_arr = ['$\epsilon$\'', 'm\'', '$\epsilon$\"', 'm\"']

ax_re.set_xlim([0, 1])
ax_im.set_xlim([0, 1])
# ax_re.set_ylim([1, 3.5])
# ax_im.set_ylim([0, 6.e-4])
ax_re.set_xlabel("f воды", fontsize=10)
ax_im.set_xlabel("f воды", fontsize=10)
ax_re.set_ylabel('$\epsilon$\'', fontsize=10)
ax_im.set_ylabel('$\epsilon$\"', fontsize=10)


ax_re.legend(leg, loc='upper left', fontsize=10)
ax_re.grid(True, which='both')
ax_im.legend(leg, loc='upper left', fontsize=10)
ax_im.grid(True, which='both')

plt.show()