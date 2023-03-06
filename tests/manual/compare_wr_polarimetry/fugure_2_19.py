#!/usr/bin/env python3
"""
Скрипт для проверки работы функция расчета диэлектирческой постоянной и индекса рефракции для воды при помоши
повторения графика 2.19 из учебника Weather Radar Polarimetry авторства Guifu Zhang
"""
from scattering_simulation.calculation.particle_properties.dielectric_permittivity.lossy_medium \
    import get_water_dielectric_constant as func_w, get_water_reflection_index as func_wid
import matplotlib.pyplot as plt
import numpy as np

# Объявлеие входных данных и вспомогательной информации для отрисовки
wavelengths = np.arange(1e-3, 1, 0.001)
temp = [-15, -5, 0, 20]
style = ['k-.','k--', 'k', 'k:']
x_limits_arr = [[1e-3, 1], [1e-3, 1], [1e-3, 1], [1e-3, 1]]
y_limits_arr = [[1, 1e2], [1, 2e1], [1, 1e2], [1e-1, 1e1]]
y_labels_arr = ['$\epsilon$\'', 'm\'', '$\epsilon$\"', 'm\"']

# Создаем фигуру с осями
fig, ((ax_w_re, ax_w_im), (ax_refr_w_re, ax_refr_w_im)) = plt.subplots(2, 2)
axes_arr = [ax_w_re, ax_refr_w_re, ax_w_im, ax_refr_w_im]

leg = []
# Отрисовка графиков для 3 темперератур
for step in range(4):
    ax_w_re.plot(wavelengths, func_w(wavelengths, temp[step]).real, style[step])
    ax_w_im.plot(wavelengths, func_w(wavelengths, temp[step]).imag, style[step])
    ax_refr_w_re.plot(wavelengths, func_wid(wavelengths, temp[step]).real, style[step])
    ax_refr_w_im.plot(wavelengths, func_wid(wavelengths, temp[step]).imag, style[step])
    leg.append(f'T = {temp[step]}°C')

for step in range(4):
    axes_arr[step].set_yscale('log')
    axes_arr[step].set_xscale('log')
    axes_arr[step].set_xlim(x_limits_arr[step])
    axes_arr[step].set_ylim(y_limits_arr[step])
    axes_arr[step].set_ylabel(y_labels_arr[step], fontsize=14)
    axes_arr[step].set_xlabel('$\lambda$, м.', fontsize=14)

ax_w_re.legend(leg, loc='lower right', fontsize=10)
ax_w_re.grid(True, which='both')

ax_w_im.legend(leg, loc='lower center', fontsize=10)
ax_w_im.grid(True, which='both')

ax_refr_w_re.legend(leg, loc='lower right', fontsize=10)
ax_refr_w_re.grid(True, which='both')

ax_refr_w_im.legend(leg, loc='lower center', fontsize=10)
ax_refr_w_im.grid(True, which='both')
# Отображение в полноэкранном режиме
mng = plt.get_current_fig_manager()
#mng.resize(*mng.window.maxsize())
mng.window.showMaximized()
plt.show()
