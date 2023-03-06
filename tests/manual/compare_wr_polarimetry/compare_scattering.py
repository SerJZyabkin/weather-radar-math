#!/usr/bin/env python3
"""
Скрипт для
"""
import numpy as np
import matplotlib.pyplot as plt
from scattering_simulation.calculation.tmatrix.spheroid import SingleSpheroidModel


def snow_scattering(_temp_c: float, _wavelengths: list):
    # Генерация данных
    xvals = np.arange(0.00005, 0.0312, 0.001)
    fig, axes = plt.subplots(len(_wavelengths), 2)
    for wavelength_id in range(len(_wavelengths)):
        ws_model = SingleSpheroidModel('wet_snow', _wavelengths[wavelength_id], 'fixed', 0.01)
        ds_model = SingleSpheroidModel('dry_snow', _wavelengths[wavelength_id], 'fixed', 0.01)
        yvals_ws_s11 = []
        yvals_ws_s22 = []
        yvals_ds_s11 = []
        yvals_ds_s22 = []
        for d_eq in xvals:
            (s11, s12), (s21, s22) = ws_model.get_scattering_matrix(d_eq, 0, 0, _temp_c)
            yvals_ws_s11.append(s11)
            yvals_ws_s22.append(s22)

            (s11, s12), (s21, s22) = ds_model.get_scattering_matrix(d_eq, 0, 0, _temp_c)
            yvals_ds_s11.append(s11)
            yvals_ds_s22.append(s22)
            print('done', d_eq)

        axes[wavelength_id][0].plot(xvals * 1000, np.abs(yvals_ws_s11), 'k')
        axes[wavelength_id][0].plot(xvals * 1000, np.abs(yvals_ws_s22), 'k--')
        axes[wavelength_id][0].plot(xvals * 1000, np.abs(yvals_ds_s11), 'r')
        axes[wavelength_id][0].plot(xvals * 1000, np.abs(yvals_ds_s22), 'r--')
        axes[wavelength_id][0].set_ylim([1.e-3, 100])
        axes[wavelength_id][0].set_xlim([0, 30])
        axes[wavelength_id][0].legend(['Shh, мокрый снег', 'Svv, мокрый снег', 'Shh, сухой снег', 'Svv, сухой снег'])
        axes[wavelength_id][0].set_yscale('log')
        axes[wavelength_id][0].grid(True)

        axes[wavelength_id][1].plot(xvals * 1000, np.rad2deg(np.angle(yvals_ws_s11)), 'k')
        axes[wavelength_id][1].plot(xvals * 1000, np.rad2deg(np.angle(yvals_ws_s22)), 'k--')
        axes[wavelength_id][1].plot(xvals * 1000, np.rad2deg(np.angle(yvals_ds_s11)), 'r')
        axes[wavelength_id][1].plot(xvals * 1000, np.rad2deg(np.angle(yvals_ds_s22)), 'r--')
        axes[wavelength_id][1].set_ylim([0, 120])
        axes[wavelength_id][1].set_xlim([0, 30])
        axes[wavelength_id][1].legend(['Shh, мокрый снег', 'Svv, мокрый снег', 'Shh, сухой снег', 'Svv, сухой снег'])
        axes[wavelength_id][1].grid(True)

    # Отображение в полноэкранном режиме
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except:
        try:
            plt.get_current_fig_manager().resize(*plt.get_current_fig_manager().window.maxsize())
        except:
            pass

    axes[int(np.floor(len(_wavelengths)/2))][0].set_ylabel('Модуль амплитуд рассеяния S, мм')
    axes[int(np.floor(len(_wavelengths)/2))][1].set_ylabel('Фаза амплитуд рассеяния S, °')
    axes[0][0].set_xlabel('a)\n\n')
    axes[0][1].set_xlabel('б)\n\n')
    axes[1][0].set_xlabel('в)\n\n')
    axes[1][1].set_xlabel('г)\n\n')
    axes[-1][0].set_xlabel('д)\n\nЭквивалентный диаметр снежинки, мм')
    axes[-1][1].set_xlabel('e)\n\nЭквивалентный диаметр снежинки, мм')
    plt.show()


def rain_scattering(_temp_c: float, _wavelengths: list):
    # Генерация данных
    xvals = np.arange(0.00005, 0.0082, 0.0001)
    fig, axes = plt.subplots(len(_wavelengths), 2)
    for wavelength_id in range(len(_wavelengths)):
        model = SingleSpheroidModel('rain', _wavelengths[wavelength_id], 'fixed', 0.01)
        yvals_a = []
        yvals_b = []
        for d_eq in xvals:
            (s11, s12), (s21, s22) = model.get_scattering_matrix(d_eq, 0, 0, _temp_c)
            yvals_a.append(s11)
            yvals_b.append(s22)

        axes[wavelength_id][0].plot(xvals * 1000, np.abs(yvals_a), 'k')
        axes[wavelength_id][0].plot(xvals * 1000, np.abs(yvals_b), 'k--')
        axes[wavelength_id][0].set_ylim([1.e-3, 100])
        axes[wavelength_id][0].set_xlim([0, 8])
        axes[wavelength_id][0].legend(['Svv', 'Svv'])
        axes[wavelength_id][0].set_yscale('log')
        axes[wavelength_id][0].grid(True)

        axes[wavelength_id][1].plot(xvals * 1000, np.rad2deg(np.angle(yvals_a)), 'k')
        axes[wavelength_id][1].plot(xvals * 1000, np.rad2deg(np.angle(yvals_b)), 'k--')
        axes[wavelength_id][1].set_ylim([0, 120])
        axes[wavelength_id][1].set_xlim([0, 8])
        axes[wavelength_id][1].legend(['Shh', 'Svv'])
        axes[wavelength_id][1].grid(True)
    # Отображение в полноэкранном режиме
    try:
        plt.get_current_fig_manager().window.showMaximized()
    except:
        try:
            plt.get_current_fig_manager().resize(*plt.get_current_fig_manager().window.maxsize())
        except:
            pass

    axes[int(np.floor(len(_wavelengths)/2))][0].set_ylabel('Модуль амплитуд рассеяния S, мм')
    axes[int(np.floor(len(_wavelengths)/2))][1].set_ylabel('Фаза амплитуд рассеяния S, °')
    axes[0][0].set_xlabel('a)\n\n')
    axes[0][1].set_xlabel('б)\n\n')
    axes[1][0].set_xlabel('в)\n\n')
    axes[1][1].set_xlabel('г)\n\n')
    axes[-1][0].set_xlabel('д)\n\nЭквивалентный диаметр капли, мм')
    axes[-1][1].set_xlabel('e)\n\nЭквивалентный диаметр капли, мм')
    plt.show()


# rain_scattering(-2.5, [0.0325, 0.075, 0.15])
snow_scattering(-7.5, [0.0325, 0.075, 0.15])
