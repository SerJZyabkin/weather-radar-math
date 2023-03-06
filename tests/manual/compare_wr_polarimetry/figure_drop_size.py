#!/usr/bin/env python3

import numpy as np
from scattering_simulation.calculation.particle_properties.rain.drop_shape import get_shape
import matplotlib.pyplot as plt


wave_length = 0.0325
temperature = 0

drop_sizes = np.arange(0, 8, 0.05)

fig, ax = plt.subplots()
ax1 = ax.twinx()

vals = np.array([get_shape(drop_size) for drop_size in drop_sizes]).transpose()

a =  vals[0]
b =  vals[1]

ax1.plot(drop_sizes, a, 'k-.')
ax1.plot(drop_sizes, b, 'k')
ax.plot(drop_sizes, np.divide(vals[1], vals[0]), 'k--')

# ax_re.set_yscale('log')
# ax_im.set_yscale('log')
# x_limits_arr = [[1e-3, 1], [1e-3, 1], [1e-3, 1], [1e-3, 1]]
# y_limits_arr = [[1, 1e2], [1, 2e1], [1, 1e2], [1e-1, 1e1]]
# y_labels_arr = ['$\epsilon$\'', 'm\'', '$\epsilon$\"', 'm\"']
#
ax1.set_xlim([0, 8])
ax1.set_xlim([0, 8])
ax1.set_ylim([0, 6])
ax.set_ylim([0, 1.2])
ax.set_xlabel("Эквивалетный размер капли, мм", fontsize=10)
ax.set_ylabel('Отношение сторон γ', fontsize=10)
ax1.set_ylabel('Размер полуосей капли, мм', fontsize=10)
#

ax1.legend(['Большая полуось \nсфероида капли', 'Малая полуось \nсфероида капли'], loc='upper right', fontsize=10)
ax.legend(['Отношение полуосей \nсфероида '], loc='upper left', fontsize=10)
ax.grid(True, which='both')



plt.show()