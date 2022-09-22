#!/usr/bin/env python3
"""
Скрипт для
"""
import numpy as np
import matplotlib.pyplot as plt
from scattering_simulation.calculation.tmatrix.spheroid import SingleSpheroidModel


def snow_zdr(_temp_c: float, _wavelengths: list, legend_strings: list):
    # Генерация данных
    colors_ws = ['k', 'k--', 'k-.', 'k:']
    colors_ds = ['r', 'r--', 'r-.', 'r:']
    xvals = np.arange(0.00005, 0.0321, 0.0003)
    fig, axes = plt.subplots(1, 1)
    for wavelength_id in range(len(_wavelengths)):
        model_ws = SingleSpheroidModel('wet_snow_fixed', _wavelengths[wavelength_id], 0.01)
        model_ds = SingleSpheroidModel('dry_snow_fixed', _wavelengths[wavelength_id], 0.01)
        yvals_ws = []
        yvals_ds = []
        for d_eq in xvals:
            print('done', wavelength_id, d_eq)
            (s11, s12), (s21, s22) = model_ws.get_scattering_matrix(d_eq, 0, 0, _temp_c)
            yvals_ws.append(s11 / s22)
            (s11, s12), (s21, s22) = model_ds.get_scattering_matrix(d_eq, 0, 0, _temp_c)
            yvals_ds.append(s11 / s22)

        axes.plot(xvals * 1000, 20 * np.log10(np.abs(yvals_ws)), colors_ws[wavelength_id])
        axes.plot(xvals * 1000, 20 * np.log10(np.abs(yvals_ds)), colors_ds[wavelength_id])

    axes.set_ylim([-2, 4])
    axes.set_xlim([0, 30])
    axes.grid(True)
    axes.legend(legend_strings)
    axes.set_ylabel('Дифференциальная отражаемость, dB')
    axes.set_xlabel('Эквивалентный диаметр капли, мм')

    # Отображение в полноэкранном режиме
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except:
        try:
            plt.get_current_fig_manager().resize(*plt.get_current_fig_manager().window.maxsize())
        except:
            pass
    plt.show()


def rain_zdr(_temp_c: float, _wavelengths: list, legend_strings: list):
    # Генерация данных
    colors = ['k', 'k--', 'k-.', 'k:']
    xvals = np.arange(0.00005, 0.0082, 0.0002)
    fig, axes = plt.subplots(1, 1)
    for wavelength_id in range(len(_wavelengths)):
        model = SingleSpheroidModel('light_rain', _wavelengths[wavelength_id], 0.01)
        yvals = []
        for d_eq in xvals:
            (s11, s12), (s21, s22) = model.get_scattering_matrix(d_eq, 0, 0, _temp_c)
            yvals.append(s11 / s22)

        axes.plot(xvals * 1000, 20 * np.log10(np.abs(yvals)), colors[wavelength_id])

    axes.set_ylim([-1, 10])
    axes.set_xlim([0, 8])
    axes.grid(True)
    axes.legend(legend_strings)
    axes.set_ylabel('Дифференциальная отражаемость, dB°')
    axes.set_xlabel('Эквивалентный диаметр капли, мм')

    # Отображение в полноэкранном режиме
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except:
        try:
            plt.get_current_fig_manager().resize(*plt.get_current_fig_manager().window.maxsize())
        except:
            pass
    plt.show()


def rain_zdr_with_temperature(_wavelength: float, _temperatures: list):
    # Генерация данных
    legend_strings = []
    colors = ['k', 'k--', 'k-.', 'k:']
    xvals = np.arange(0.00005, 0.0082, 0.0002)
    model = SingleSpheroidModel('light_rain', _wavelength, 0.01)
    fig, axes = plt.subplots(1, 1)
    for temp_id in range(len(_temperatures)):
        yvals = []
        for d_eq in xvals:
            (s11, s12), (s21, s22) = model.get_scattering_matrix(d_eq, 0, 0, _temperatures[temp_id])
            yvals.append(s11 / s22)

        axes.plot(xvals * 1000, 20 * np.log10(np.abs(yvals)), colors[temp_id])
        legend_strings.append(f'T = {_temperatures[temp_id]} °C.')

    axes.set_ylim([-1, 5])
    axes.set_xlim([0, 8])
    axes.grid(True)
    axes.legend(legend_strings)
    axes.set_ylabel('Дифференциальная отражаемость, dB')
    axes.set_xlabel('Эквивалентный диаметр капли, мм')

    # Отображение в полноэкранном режиме
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except:
        try:
            plt.get_current_fig_manager().resize(*plt.get_current_fig_manager().window.maxsize())
        except:
            pass
    plt.show()


# rain_zdr(-7.5, [0.0325, 0.07, 0.15], ['Х-диапазон, 3.25 см длина волны', 'С-диапазон, 7 см длина волны',
#                                       'S-диапазон, 15 см длина волны'])
# rain_zdr_with_temperature(0.0325, [-10, -5, 0, 5])
snow_zdr(-7.5, [0.0325, 0.07, 0.15], ['Мокрый снег, Х-диапазон, 3.25 см длина волны',
                                      'Сухой снег, Х-диапазон, 3.25 см длина волны',
                                      'Мокрый снег, С-диапазон, 7 см длина волны',
                                      'Сухой снег, С-диапазон, 7 см длина волны',
                                      'Мокрый снег, S-диапазон, 15 см длина волны',
                                      'Сухой снег, S-диапазон, 15 см длина волны'])
