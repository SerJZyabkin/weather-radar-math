#!/usr/bin/env python3
from .ampld_lp import calculate
import numpy as np
from dataclasses import dataclass
from ..common_expressions import UniformRandomValue, GaussRandomValue
from random import seed
from abc import ABC, abstractmethod
from typing import Union, Callable, Iterator, Tuple
from ..particle_properties.rain.drop_size import factory_exponential_distribution
from ..particle_properties.rain.drop_shape import get_gamma_rain
from ..particle_properties.dielectric_permittivity.lossy_medium import get_water_dielectric_constant, get_water_reflection_index
from ..rayleigh_approximation.spheroid_scattering import backscattering_amplitude
import matplotlib.pyplot as plt
from .spheroid_info import *
from scipy.integrate import quad
import matplotlib.pyplot as plt
from random import random
import os


def generator_temperature(temperature: (float, float), init_seed: float = None):
    generator = UniformRandomValue(*temperature, init_seed)
    while True:
        yield next(generator)


class SingleSpheroidModel:
    def __init__(self, class_type: str, wavelength: float, mode: str = 'general', init_seed: float = None):
        self.wavelength = wavelength
        self.init_seed = init_seed

        # Получение модели характеристик сфероида, избрание фабрики для выбранного класса
        if class_type in ('drizzle', 'medium_rain', 'heavy_rain', 'large_drops', 'dry_snow',
                          'wet_snow', 'ice_crystals') and mode == 'general':
            model_spheroid = get_spheroid_info(class_type, init_seed)
        elif class_type in ('rain', 'dry_snow', 'wet_snow') and mode == 'fixed':
            model_spheroid = get_fixed_spheroid_info(class_type, init_seed)
        elif class_type in ('drizzle', 'rain', 'dry_snow', 'wet_snow', 'ice_crystals') and mode == 'compare':
            model_spheroid = get_spheroid_comparation(class_type, init_seed)
        else:
            print(f'SingleSpheroidModel.__init__: Неизвестный класс {class_type} получен в конструкторе.')        #

        # Получение функциональных зависимостей
        self.dimensions_model = model_spheroid.get_dimensions_model()
        self.permittivity_model = model_spheroid.get_permittivity_model()

    def get_scattering_matrix(self, equdiameter: float, tilt_angle: float, falling_angle: float,
                              temperature: float) -> ((complex, complex), (complex, complex)):
        epsilon_permittivity = self.permittivity_model(self.wavelength, temperature)
        gamma_dimensions = self.dimensions_model(equdiameter)

        s_out = calculate(self.wavelength, equdiameter, epsilon_permittivity, gamma_dimensions, 90, tilt_angle,
                          [90 - falling_angle, 90 - falling_angle, 0, 0])
        # print(self.wavelength, equdiameter, epsilon_permittivity, gamma_dimensions, az_angle, 90. + el_angle,
        #                   [0] * 4)
        return (s_out[0] * 1000, s_out[1] * 1000), (s_out[2] * 1000, s_out[3] * 1000)


class SpheroidModel:
    def __init__(self, class_type: str, wavelength: float, mode: str = 'general', init_seed: float = None):
        self.wavelength = wavelength
        self.init_seed = init_seed

        # Получение модели характеристик сфероида, избрание фабрики для выбранного класса
        if class_type in ('drizzle', 'medium_rain', 'heavy_rain', 'large_drops', 'dry_snow',
                          'wet_snow', 'ice_crystals') and mode == 'general':
            self.model_info = get_spheroid_info(class_type, init_seed)
        elif class_type in ('rain', 'dry_snow', 'wet_snow') and mode == 'fixed':
            self.model_info = get_fixed_spheroid_info(class_type, init_seed)
        elif class_type in ('drizzle', 'rain', 'dry_snow', 'wet_snow', 'ice_crystals') and mode == 'compare':
            self.model_info = get_spheroid_comparation(class_type, init_seed)
        else:
            print(f'SpheroidModel.__init__: Неизвестный класс {class_type} получен в конструкторе.')

        # Получение функциональных зависимостей
        self.dimensions_model = self.model_info.get_dimensions_model()
        self.permittivity_model = self.model_info.get_permittivity_model()
        self.single_spheroid_model = SingleSpheroidModel(class_type, wavelength, mode, init_seed)

        # Получение констанстных значений
        self.min_diameter, self.max_diameter = self.model_info.get_min_diameter(), self.model_info.get_max_diameter()

    def get_distribution_by_diams(self):
        distribution_model, num_bins_distribution = self.model_info.get_distribution_model()
        out_data = np.zeros([2, num_bins_distribution])
        delta_diameters = (self.max_diameter - self.min_diameter) / num_bins_distribution
        diameter_limits = np.linspace(self.min_diameter, self.max_diameter, num_bins_distribution + 1)
        out_data[0, :] = (diameter_limits[:-1] + delta_diameters / 2) / 1000
        for step in range(num_bins_distribution):
            out_data[1, step] = int(np.round(quad(distribution_model, diameter_limits[step],
                                                  diameter_limits[step + 1])[0]))
        return out_data

    def get_products(self, _temperature, falling_angle: float = 0.):
        while True:
            diam_distribution = self.get_distribution_by_diams()
            if sum(diam_distribution[1, :]) > 0:
                break
            else:
                # print(diam_distribution[1, :])
                pass

        accum_s11 = 0
        accum_s22 = 0
        accum_s12 = 0
        accum_corr = 0
        accum_dp = 0

        tilt_model = self.model_info.get_tilt_model()
        for step in range(len(diam_distribution[0, :])):
            (s11, s21), (s12, s22) = self.single_spheroid_model.get_scattering_matrix(diam_distribution[0, step],
                                                    next(tilt_model), falling_angle, _temperature)
            accum_s11 += diam_distribution[1, step] * np.abs(np.square(s11))
            accum_s22 += diam_distribution[1, step] * np.abs(np.square(s22))
            accum_s12 += diam_distribution[1, step] * np.abs(np.square(s21))
            accum_corr += diam_distribution[1, step] * (np.conj(s11) * s22)
            accum_dp += diam_distribution[1, step] * (s11 - s22)

        Ze = 4 * np.power(self.wavelength * 1000. / np.pi, 4.) / 0.92 * accum_s11
        Zh = 10 * np.log10(Ze) - 18

        Zdr = 10 * np.log10(accum_s11 / accum_s22)

        LDR = 10 * np.log10(accum_s12 / accum_s11)

        ro_hv = np.abs(accum_corr) / np.sqrt(accum_s22) / np.sqrt(accum_s11)

        Kdp = 180 / np.pi * self.wavelength * np.real(accum_dp)

        return Zh, Zdr, LDR, ro_hv, Kdp

def plot_all():
    with open('ice_crystals.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_ic = np.frombuffer(raw_bytes, dtype=np.float)
        data_ic = np.array(data_ic.reshape([int(len(data_ic) / 5), 5]))
        for step in range(data_ic.shape[0]):
            data_ic[step, 0] = data_ic[step, 0] + random() * 6 - 10
        fid.close()

    with open('dry_snow.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_ds = np.frombuffer(raw_bytes, dtype=np.float)
        data_ds = np.array(data_ds.reshape([int(len(data_ds) / 5), 5]))
        fid.close()

    with open('wet_snow.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_ws = np.frombuffer(raw_bytes, dtype=np.float)
        data_ws = np.array(data_ws.reshape([int(len(data_ws) / 5), 5]))
        for step in range(data_ws.shape[0]):
            data_ws[step, 0] = data_ws[step, 0] #+ random() * 6 - 10
        fid.close()

    with open('rain.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_r = np.frombuffer(raw_bytes, dtype=np.float)
        data_r = np.array(data_r.reshape([int(len(data_r) / 5), 5]))
        for step in range(data_r.shape[0]):
            data_r[step, 0] = data_r[step, 0] #+ random() * 6 - 10
        fid.close()

    with open('drizzle.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_drizzle = np.frombuffer(raw_bytes, dtype=np.float)
        data_drizzle = data_drizzle.reshape([int(len(data_drizzle) / 5), 5])
        fid.close()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(data_ic[:, 0], data_ic[:, 1], '+', color='lightblue')
    ax[1].plot(data_ic[:, 0], data_ic[:, 2], '+', color='lightblue')

    ax[0].plot(data_ds[:, 0], data_ds[:, 1], '*', color='green')
    ax[1].plot(data_ds[:, 0], data_ds[:, 2], '*', color='green')

    ax[0].plot(data_ws[:, 0], data_ws[:, 1], '+', color='darkblue')
    ax[1].plot(data_ws[:, 0], data_ws[:, 2], '+', color='darkblue')

    ax[0].plot(data_r[:, 0], data_r[:, 1], 'o', color='red')
    ax[1].plot(data_r[:, 0], data_r[:, 2], 'o', color='red')

    ax[0].plot(data_drizzle[:, 0], data_drizzle[:, 1], 'o', color='yellow')
    ax[1].plot(data_drizzle[:, 0], data_drizzle[:, 2], 'o', color='yellow')


    ax[0].legend(['Кристаллы льда', 'Сухой снег', 'Мокрый снег', 'Дождь', 'Морось'])
    ax[1].legend(['Кристаллы льда', 'Сухой снег', 'Мокрый снег', 'Дождь', 'Морось'])
    ax[0].set_xlim([-70, 70])
    ax[0].set_ylim([-1, 5])
    ax[0].grid(True)
    ax[1].set_xlim([-10, 70])
    ax[1].set_ylim([-50, -10])
    ax[1].grid(True)
    plt.show()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(data_ic[:, 0], data_ic[:, 3], '+', color='lightblue')
    ax[1].plot(data_ic[:, 0], data_ic[:, 4], '+', color='lightblue')
    ax[0].plot(data_ds[:, 0], data_ds[:, 3], '*', color='green')
    ax[1].plot(data_ds[:, 0], data_ds[:, 4], '*', color='green')

    ax[0].plot(data_ws[:, 0], data_ws[:, 3], '+', color='darkblue')
    ax[1].plot(data_ws[:, 0], data_ws[:, 4], '+', color='darkblue')

    ax[0].plot(data_r[:, 0], data_r[:, 3], 'o', color='red')
    ax[1].plot(data_r[:, 0], data_r[:, 4], 'o', color='red')

    ax[0].plot(data_drizzle[:, 0], data_drizzle[:, 3], 'o', color='yellow')
    ax[1].plot(data_drizzle[:, 0], data_drizzle[:, 4], 'o', color='yellow')


    plt.legend(['Кристаллы льда', 'Сухой снег', 'Мокрый снег', 'Дождь', 'Морось'])
    ax[0].set_xlim([-10, 70])
    ax[0].set_ylim([0.975, 1.005])
    ax[0].grid(True)
    ax[1].set_xlim([-10, 70])
    ax[1].set_ylim([0, 15])
    ax[1].grid(True)
    plt.show()


def calculate_all():
    data_ic = get_for_compare_ice_crystals()
    with open('ice_crystals.bin', 'wb') as fid:
        fid.write(data_ic.tobytes())
        fid.close()
    #
    # data_ws = get_for_compare_wet_snow()
    # with open('wet_snow.bin', 'wb') as fid:
    #     fid.write(data_ws.tobytes())
    #     fid.close()
    #
    # data_rain = get_for_compare_rain()
    # with open('rain.bin', 'wb') as fid:
    #     fid.write(data_rain.tobytes())
    #     fid.close()
    #
    # data_ds = get_for_compare_dry_snow()
    # with open('dry_snow.bin', 'wb') as fid:
    #     fid.write(data_ds.tobytes())
    #     fid.close()
    #
    # data_drizzle = get_for_compare_drizzle()
    # with open('drizzle.bin', 'wb') as fid:
    #     fid.write(data_drizzle.tobytes())
    #     fid.close()


def get_for_compare_drizzle(init_seed: float = None) -> np.array:
    wavelength = 0.15
    min_temp = -15.
    max_temp = 5.
    num_iters = 1000

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_drizzle = SpheroidModel('drizzle', wavelength, 'compare', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_drizzle.get_products(next(t_gen), 7 - 4 * random()))
        print('drz', step)
    out_data = np.array(out_data)
    return out_data


def get_for_compare_rain(init_seed: float = None) -> np.array:
    wavelength = 0.15
    min_temp = -5.
    max_temp = 5.
    num_iters = 1

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('rain', wavelength, 'compare', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 14 - 6 * random()))
        print('r', step)
    out_data = np.array(out_data)
    return out_data


def get_for_compare_wet_snow(init_seed: float = None) -> np.array:
    wavelength = 0.15
    min_temp = -15.
    max_temp = 5.
    num_iters = 1000

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('wet_snow', wavelength, 'compare', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 14 - 6 * random()))
        print('ws', step)
    out_data = np.array(out_data)
    return out_data


def get_for_compare_dry_snow(init_seed: float = None) -> np.array:
    wavelength = 0.15
    min_temp = -15.
    max_temp = 0.
    num_iters = 1

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('dry_snow', wavelength, 'compare', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 14 - 6 * random()))
        print('ds', step)
    out_data = np.array(out_data)
    return out_data


def get_for_compare_ice_crystals(init_seed: float = None) -> np.array:
    wavelength = 0.15
    min_temp = -15.
    max_temp = 0.
    num_iters = 1000

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('ice_crystals', wavelength, 'compare', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 6 - 2 * random()))
        print('ic', step)
    out_data = np.array(out_data)
    return out_data



def main():
    path_to_data = 'E:/Git/weather-radar/weather-radar-math/bin' # '/media/serj/Data/Git/repos/weather-radar-math/bin'
    if not os.path.exists(path_to_data):
        os.mkdir(path_to_data)

    os.chdir(path_to_data)

    calculate_all()
    plot_all()
    quit()
    # xxx = [0.0325, 0.0075, 2.533353951788893 + 0.05540354805478787j, 0.7637107722162353]
    #
    # xx = get_spheroid_info('dry_snow_fixed', 0.01).get_permittivity_model()(0.0325, -5)
    # xx1 = get_spheroid_info('wet_snow_fixed', 0.01).get_permittivity_model()(0.0325, -5)
    # xx3 = get_spheroid_info('drizzle', 0.01).get_permittivity_model()(0.0325, -5)
    # print(xx, xx1, xx3)
    #
    # def do_thing(a1, eps):
    #     s11, s12, s21, s22 = calculate(0.0325, 0.01, eps, 0.5,
    #                                    90, a1, [95, 95, 0, 0])
    #
    #     if np.abs(s21) > 0:
    #         print(20 * np.log10(np.abs(s11) / np.abs(s22)), 20 * np.log10(np.abs(s12) / np.abs(s11)),
    #               20 * np.log10(np.abs(s21) / np.abs(s22)), np.abs(s11 * np.conj(s22)) / np.sqrt(np.abs(np.square(s11)) * np.abs(np.square(s22))))
    #     else:
    #         print(20 * np.log10(np.abs(s11) / np.abs(s22)), -180)
    #
    # do_thing(2, xx)
    # do_thing(2, xx1)
    # do_thing(1, xx3)
    # # do_thing(0, 180)
    # quit()
    TEMP = 10
    WAVELENGTH = 0.15
    # xx = get_spheroid_info('dry_snow_fixed', 0.01).get_permittivity_model()(0.0325, -5)
    # xx1 = get_spheroid_info('wet_snow_fixed', 0.01).get_permittivity_model()(0.0325, -5)
    # xx3 = get_spheroid_info('drizzle', 0.01).get_permittivity_model()(0.0325, -5)
    # print(xx, xx1, xx3)
    #
    # ssr = SingleSpheroidModel('drizzle', 0.0325, 0.01).get_scattering_matrix(0.001, 0, 0, -5)
    # ssds = SingleSpheroidModel('dry_snow_fixed', 0.0325, 0.01).get_scattering_matrix(0.001, 0, 0, -5)
    # ssws = SingleSpheroidModel('wet_snow_fixed', 0.0325, 0.01).get_scattering_matrix(0.001, 0, 0, -5)


    # model_rain = SpheroidModel('rain_fixed', WAVELENGTH, 0.01)
    #
    # conc = model_rain.get_total_concentration()
    # print(conc)
    # print(model_rain.get_products(TEMP))

    model_ws = SpheroidModel('wet_snow', WAVELENGTH, 'general', 0.1)
    print(*model_ws.get_products(TEMP))

    model_ws = SpheroidModel('dry_snow', WAVELENGTH, 'general', 0.1)
    print(*model_ws.get_products(TEMP))

    model_ws = SpheroidModel('medium_rain', WAVELENGTH, 'general', 0.1)
    print(*model_ws.get_products(TEMP))

    model_ws = SpheroidModel('heavy_rain', WAVELENGTH, 'general', 0.1)
    print(*model_ws.get_products(TEMP))

    quit()
    num_iters = 70
    model_ws = SpheroidModel('wet_snow', WAVELENGTH, 0.01)
    data = np.zeros([2, num_iters])
    for step in range(num_iters):
        data[0, step],  data[1, step] = model_ws.get_products(TEMP)
    print('done ws')
    plt.plot(data[0, :], data[1, :], 'c*')

    ds_model = SpheroidModel('dry_snow', WAVELENGTH, 0.01)
    for step in range(num_iters):
        data[0, step],  data[1, step] = ds_model.get_products(TEMP)
    plt.plot(data[0, :], data[1, :], 'r*')
    print('done ds')

    num_iters = 40
    data = np.zeros([2, num_iters])
    lr_model = SpheroidModel('light_rain', WAVELENGTH, 0.01)
    for step in range(num_iters):
        data[0, step],  data[1, step] = lr_model.get_products(TEMP)
    plt.plot(data[0, :], data[1, :], 'yo')
    print('done lr')
    mr_model = SpheroidModel('medium_rain', WAVELENGTH, 0.01)
    for step in range(num_iters):
        data[0, step],  data[1, step] = mr_model.get_products(TEMP)
    plt.plot(data[0, :], data[1, :], 'bo')

    print('done mr')
    num_iters = 20
    data = np.zeros([2, num_iters])
    hr_model = SpheroidModel('heavy_rain', WAVELENGTH, 0.01)
    for step in range(num_iters):
        data[0, step],  data[1, step] = hr_model.get_products(TEMP)
    plt.plot(data[0, :], data[1, :], 'ko')

    num_iters = 40
    data = np.zeros([2, num_iters])
    for step in range(num_iters):
        data[0, step] = uniform(5, 30)
        data[1, step] = uniform(-1.5, 2.5)
    plt.plot(data[0, :], data[1, :], 'g+')


    plt.gca().set_ylim([-2, 7])
    plt.gca().set_xlim([0, 70])
    plt.grid(True)
    plt.legend(['Мокрый снег', 'Сухой снег', 'Капли слабого дождя',
                'Капли умеренного дождя', 'Капли ливневого дождя', 'Кристаллы льда'])
    plt.gca().set_ylabel('Дифференциальная отражаемость, dB')
    plt.gca().set_xlabel('Радиолокационная отражаемость, dBZ')

    plt.show()

