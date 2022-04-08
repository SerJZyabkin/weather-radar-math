#!/usr/bin/env python3
"""
Скрипт для проверки работы функция расчета диэлектирческой постоянной и индекса рефракции для льда при помоши
повторения графика, аналогичного графику 2.20 из учебника Weather Radar Polarimetry авторства Guifu Zhang
"""
from scattering_simulation.calculation.dielectric_permittivity.lossy_medium \
    import get_ice_dielectric_constant as func_i, get_ice_reflection_index as func_rid
import matplotlib.pyplot as plt
import numpy as np

# Объявлеие входных данных и вспомогательной информации для отрисовки
wavelengths = np.arange(1e-3, 1, 0.001)
temp = [-16, -8, 0]
style = ['k-.', 'k--', 'k']
x_limits_arr = [[1e-3, 1], [1e-3, 1], [1e-3, 1], [1e-3, 1]]
y_limits_arr = [[0, 5], [0, 5], [1e-5, 1e-2], [1e-5, 1e-2]]
y_labels_arr = ['$\epsilon$\'', 'm\'', '$\epsilon$\"', 'm\"']
y_scales_arr = ['linear', 'linear', 'log', 'log']

# Создаем фигуру с осями
fig, ((ax_w_re, ax_w_im), (ax_refr_w_re, ax_refr_w_im)) = plt.subplots(2, 2)
axes_arr = [ax_w_re, ax_refr_w_re, ax_w_im, ax_refr_w_im]

# Отрисовка графиков для 3 темперератур
for step in range(3):
    ax_w_re.plot(wavelengths, func_i(wavelengths, temp[step]).real, style[step])
    ax_w_im.plot(wavelengths, func_i(wavelengths, temp[step]).imag, style[step])
    ax_refr_w_re.plot(wavelengths, func_rid(wavelengths, temp[step]).real, style[step])
    ax_refr_w_im.plot(wavelengths, func_rid(wavelengths, temp[step]).imag, style[step])

for step in range(4):
    axes_arr[step].set_yscale(y_scales_arr[step])
    axes_arr[step].set_xscale('log')
    axes_arr[step].set_xlim(x_limits_arr[step])
    axes_arr[step].set_ylim(y_limits_arr[step])
    axes_arr[step].set_ylabel(y_labels_arr[step], fontsize=14)
    axes_arr[step].set_xlabel('$\lambda$, м.', fontsize=14)

ax_w_re.legend(['T = -16 °C', 'T = -8 °C', 'T = 0 °C'], loc='lower right', fontsize=14)

# Отображение в полноэкранном режиме
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()
