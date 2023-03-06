#!/usr/bin/env python3
"""
Скрипт для
"""
import numpy as np
import matplotlib.pyplot as plt
from scattering_simulation.calculation.tmatrix.spheroid import SingleSpheroidModel, get_fixed_spheroid_info


def rain_reflecitivity(_temp: float, _wavelengths: list, _equdiameters: list):
    # Генерация данных
    xvals = np.arange(10, 1e5 + 1, 100)
    fig, axes = plt.subplots(1, len(_wavelengths))
    lines_style = ['k-', 'k:', 'k-.', 'k--']
    for wavelength_id in range(len(_wavelengths)):
        model = SingleSpheroidModel('rain', _wavelengths[wavelength_id], 'fixed', 0.01)
        diel_epsi = get_fixed_spheroid_info('rain', 0.01).get_permittivity_model()(_wavelengths[wavelength_id], _temp)
        K2 = np.square(np.abs((diel_epsi * diel_epsi - 1) / (diel_epsi * diel_epsi + 2)))
        legends_list = []
        for d_eq, line_style in zip(_equdiameters, lines_style):
            yvals = []
            (s11, _), _ = model.get_scattering_matrix(d_eq / 1000., 0, 0, _temp)
            multi = 4 * np.power(_wavelengths[wavelength_id] / np.pi, 4.) / K2 * np.square(np.abs(s11)) * 1e6
            for xval in xvals:
                Ze = xval * multi
                yvals.append(10 * np.log10(Ze))

            axes[wavelength_id].plot(xvals, yvals, line_style)
            legends_list.append(f'Средний размер капли {d_eq} мм.')

        axes[wavelength_id].set_xlim([min(xvals), max(xvals)])
        axes[wavelength_id].set_ylim([-5, 50])
        axes[wavelength_id].set_yticks(np.arange(-5, 50, 5))
        axes[wavelength_id].legend(legends_list)
        axes[wavelength_id].set_xscale('log')
        axes[wavelength_id].grid(True)

    try:
        plt.get_current_fig_manager().window.showMaximized()
    except:
        try:
            plt.get_current_fig_manager().resize(*plt.get_current_fig_manager().window.maxsize())
        except:
            pass

    axes[0].set_xlabel('\n\na)')
    axes[2].set_xlabel('\n\nв)')
    axes[int(np.floor(len(_wavelengths)/2))].set_xlabel('Концентрация капель дождя, */м^3 \n\nб)')
    axes[0].set_ylabel('Радиолокационная отражаемость, dbZ')

    plt.show()


def snow_reflectivity(_temp: float, _wavelength: float, _equdiameters: list):
    # Генерация данных
    xvals = np.arange(10, 1e5 + 1, 100)
    fig, axes = plt.subplots(1, 1)
    lines_style = ['-', ':', '-.', '--', '-', ':', 'densely dashed']

    model_ws = SingleSpheroidModel('wet_snow', _wavelength, 'fixed', 0.01)
    model_ds = SingleSpheroidModel('dry_snow', _wavelength, 'fixed', 0.01)
    diel_epsi_ws = get_fixed_spheroid_info('wet_snow', 0.01).get_permittivity_model()(_wavelength, _temp)
    diel_epsi_ds = get_fixed_spheroid_info('dry_snow', 0.01).get_permittivity_model()(_wavelength, _temp)

    K2_ws = 0.92 #np.square(np.abs((diel_epsi_ws * diel_epsi_ws - 1) / (diel_epsi_ws * diel_epsi_ws + 2)))
    K2_ds = 0.92 #np.square(np.abs((diel_epsi_ds * diel_epsi_ds - 1) / (diel_epsi_ds * diel_epsi_ds + 2)))

    legends_list = []
    for d_eq, line_style in zip(_equdiameters, lines_style):
        yvals = []
        (s11, _), _ = model_ws.get_scattering_matrix(d_eq / 1000., 0, 0, _temp)
        multi = 4 * np.power(_wavelength / np.pi, 4.) / K2_ws * np.square(np.abs(s11)) * 1e6
        for xval in xvals:
            Ze = xval * multi
            yvals.append(10 * np.log10(Ze))
        legends_list.append(f'Мокрые снежинки среднего размера {d_eq} мм.')
        axes.plot(xvals, yvals, 'k' + line_style)

        yvals = []
        (s11, _), _ = model_ds.get_scattering_matrix(d_eq / 1000., 0, 0, _temp)
        multi = 4 * np.power(_wavelength / np.pi, 4.) / K2_ds * np.square(np.abs(s11)) * 1e6
        for xval in xvals:
            Ze = xval * multi
            yvals.append(10 * np.log10(Ze))

        axes.plot(xvals, yvals, 'r' + line_style)
        legends_list.append(f'Сухие снежинки среднего размера {d_eq} мм.')

    axes.set_xlim([min(xvals), max(xvals)])
    axes.set_ylim([-5, 50])
    axes.set_yticks(np.arange(-5, 50, 5))
    axes.legend(legends_list)
    axes.set_xscale('log')
    axes.grid(True)

    try:
        plt.get_current_fig_manager().window.showMaximized()
    except:
        try:
            plt.get_current_fig_manager().resize(*plt.get_current_fig_manager().window.maxsize())
        except:
            pass

    axes.set_xlabel('Концентрация снежинок, */м^3 \n\nб)')
    axes.set_ylabel('Радиолокационная отражаемость, dbZ')

    plt.show()


rain_reflecitivity(-5, [0.0325, 0.07, 0.14], [1, 2, 3, 4])
snow_reflectivity(-10, 0.0325, [1, 2, 3, 4, 9])
