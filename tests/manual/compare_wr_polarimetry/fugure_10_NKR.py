#!/usr/bin/env python3
"""

"""
from scattering_simulation.calculation.particle_properties.dielectric_permittivity.lossy_medium \
    import get_water_dielectric_constant as func_w
from scattering_simulation.calculation.particle_properties.dielectric_permittivity.lossy_medium \
    import get_ice_dielectric_constant as func_i
import matplotlib.pyplot as plt
import numpy as np

# Объявлеие входных данных и вспомогательной информации для отрисовки
wavelengths = np.arange(1e-3, 1, 0.001)
temp = [-15, -5, 0, 20]
style = ['k-.','k--', 'k', 'k:']
y_scale = ['log', 'linear', 'log','log']
x_limits_arr = [[1e-3, 1], [1e-3, 1], [1e-3, 1], [1e-3, 1]]
y_limits_arr = [[1, 1e2], [1, 5], [1, 1e2], [1e-5, 1e-2]]
y_labels_arr = ['$\epsilon$\' для воды', '$\epsilon$\' для льда', '$\epsilon$\" для воды', '$\epsilon$\" для льда']

# Создаем фигуру с осями
fig, ((ax_w_re, ax_w_im), (ax_i_re, ax_i_im)) = plt.subplots(2, 2)
axes_arr = [ax_w_re, ax_i_re, ax_w_im, ax_i_im]

leg = []
# Отрисовка графиков для 3 темперератур
for step in range(4):
    ax_w_re.plot(wavelengths, func_w(wavelengths, temp[step]).real, style[step])
    ax_w_im.plot(wavelengths, func_w(wavelengths, temp[step]).imag, style[step])
    if step < 3:
        ax_i_re.plot(wavelengths, func_i(wavelengths, temp[step]).real, style[step])
        ax_i_im.plot(wavelengths, func_i(wavelengths, temp[step]).imag, style[step])
    leg.append(f'T = {temp[step]}°C')

for step in range(4):
    axes_arr[step].set_yscale(y_scale[step])
    axes_arr[step].set_xscale('log')
    axes_arr[step].set_xlim(x_limits_arr[step])
    axes_arr[step].set_ylim(y_limits_arr[step])
    axes_arr[step].set_ylabel(y_labels_arr[step], fontsize=14)
    axes_arr[step].set_xlabel('$\lambda$, м.', fontsize=14)

ax_w_re.legend(leg, loc='lower right', fontsize=10)
ax_w_re.grid(True, which='both')

ax_w_im.legend(leg, loc='upper right', fontsize=10)
ax_w_im.grid(True, which='both')

ax_i_re.legend(leg[:-1], loc='lower right', fontsize=10)
ax_i_re.grid(True, which='both')

ax_i_im.legend(leg[:-1], loc='upper left', fontsize=10)
ax_i_im.grid(True, which='both')
# Отображение в полноэкранном режиме
mng = plt.get_current_fig_manager()
#mng.resize(*mng.window.maxsize())
mng.window.showMaximized()
plt.show()
