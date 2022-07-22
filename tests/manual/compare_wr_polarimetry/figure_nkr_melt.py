#!/usr/bin/env python3

import numpy as np
from scattering_simulation.calculation.particle_properties.dielectric_permittivity.particular_mixtures import \
    get_mixing_for_wet_snow
import matplotlib.pyplot as plt


wave_length = 0.0325
temperature = 0

melting_progress_arr = np.arange(0, 1, 0.001)

fig, ((ax_re, ax_im)) = plt.subplots(1, 2)

vals = np.array([get_mixing_for_wet_snow(wave_length, temperature, 0.2, melting_progress)
            for melting_progress in melting_progress_arr])
ax_re.plot(melting_progress_arr, vals.real, 'k-.')
ax_im.plot(melting_progress_arr, vals.imag, 'k-.')

vals = np.array([get_mixing_for_wet_snow(wave_length, -7, 0.2, melting_progress)
            for melting_progress in melting_progress_arr])
ax_re.plot(melting_progress_arr, vals.real, 'k--')
ax_im.plot(melting_progress_arr, vals.imag, 'k--')


vals = np.array([get_mixing_for_wet_snow(wave_length, -15, 0.2, melting_progress)
            for melting_progress in melting_progress_arr])
ax_re.plot(melting_progress_arr, vals.real, 'k')
ax_im.plot(melting_progress_arr, vals.imag, 'k')

# ax_re.set_yscale('log')
# ax_im.set_yscale('log')
# x_limits_arr = [[1e-3, 1], [1e-3, 1], [1e-3, 1], [1e-3, 1]]
# y_limits_arr = [[1, 1e2], [1, 2e1], [1, 1e2], [1e-1, 1e1]]
# y_labels_arr = ['$\epsilon$\'', 'm\'', '$\epsilon$\"', 'm\"']
#
ax_re.set_xlim([0, 1])
ax_im.set_xlim([0, 1])
# ax_re.set_ylim([1, 3.5])
# ax_im.set_ylim([0, 6.e-4])
ax_re.set_xlabel("Коэффициент таяния γ", fontsize=10)
ax_im.set_xlabel("Коэффициент таяния γ", fontsize=10)
ax_re.set_ylabel('$\epsilon$\'', fontsize=10)
ax_im.set_ylabel('$\epsilon$\"', fontsize=10)
#
leg = [f'T = {step_temp}°C' for step_temp in [5, -5, -15]]

ax_re.legend(leg, loc='upper left', fontsize=10)
ax_re.grid(True, which='both')
ax_im.legend(leg, loc='upper left', fontsize=10)
ax_im.grid(True)

plt.show()