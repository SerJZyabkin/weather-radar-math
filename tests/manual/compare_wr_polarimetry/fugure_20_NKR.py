#!/usr/bin/env python3
"""
Скрипт для проверки работы функция расчета диэлектирческой постоянной и индекса рефракции для льда при помоши
повторения графика, аналогичного графику 2.22 из учебника Weather Radar Polarimetry авторства Guifu Zhang
"""

import numpy as np
from scattering_simulation.calculation.particle_properties.dielectric_permittivity.lossy_medium import get_ice_dielectric_constant, \
    get_air_dielectric_constant
from scattering_simulation.calculation.particle_properties.dielectric_permittivity.medium_mixture import get_mixing_maxwell_garnett, \
    get_mixing_ponder_vansandern
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

fig, ((ax_re, ax_im)) = plt.subplots(1, 2)

xx = np.arange(0, 90, 1)
func_yy1 = interp1d([-30, 0, 30, 45, 60, 90, 120], [7.5, 6, 4, 0, -4, -6, -7.5], kind='cubic')
func_yy2 = interp1d([-30, 0, 30, 45, 60, 90, 120], [4.2, 3.8, 2.2, 0, -2.2, -3.8, -4.2], kind='cubic')
yy1 = func_yy1(xx)
yy2 = func_yy2(xx)
ax_re.plot(xx, yy1, 'k-')
ax_re.plot(xx, yy2, 'k--')

func_yy1 = interp1d([-40, 0, 61, 90, 120], [-80, -40, -13, -43, -80], kind='cubic')
func_yy2 = interp1d([-40, 0, 61, 90, 120], [-80, -40, -19, -43, -80], kind='cubic')
yy1 = func_yy1(xx)
yy2 = func_yy2(xx)
ax_im.plot(xx, yy1, 'k-')
ax_im.plot(xx, yy2, 'k--')

ax_re.set_xlim([0, 90])
ax_im.set_xlim([-0, 90])
ax_re.set_ylim([-7, 7])
ax_im.set_ylim([-40, 0])
ax_re.set_xlabel("Угол падения, °", fontsize=10)
ax_im.set_xlabel("Угол падения, °", fontsize=10)
ax_re.set_ylabel('Отношение амплитуд рассеяния, dB', fontsize=10)
ax_im.set_ylabel('Отношение амплитуд рассеяния, dB', fontsize=10)

leg = ['$\gamma$ = 2', '$\gamma$= 1,5']

ax_re.legend(leg, loc='lower left', fontsize=10)
ax_re.grid(True, which='both')
ax_im.legend(leg, loc='upper left', fontsize=10)
ax_im.grid(True, which='both')

plt.show()