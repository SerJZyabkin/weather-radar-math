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
from .spheroid import *


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

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(data_ic[:, 0][:100], data_ic[:, 1][:100], '+', color='lightblue')
    ax[1].plot(data_ic[:, 0][:100], data_ic[:, 2][:100], '+', color='lightblue')

    ax[0].plot(data_ds[:, 0][:100], data_ds[:, 1][:100], '*', color='green')
    ax[1].plot(data_ds[:, 0][:100], data_ds[:, 2][:100], '*', color='green')

    ax[0].plot(data_ws[:, 0][:100], data_ws[:, 1][:100], '+', color='darkblue')
    ax[1].plot(data_ws[:, 0][:100], data_ws[:, 2][:100], '+', color='darkblue')

    ax[0].plot(data_r[:, 0][:100], data_r[:, 1][:100], 'o', color='red')
    ax[1].plot(data_r[:, 0][:100], data_r[:, 2][:100], 'o', color='red')

    ax[0].plot(data_drizzle[:, 0][:100], data_drizzle[:, 1][:100], 'o', color='yellow')
    ax[1].plot(data_drizzle[:, 0][:100], data_drizzle[:, 2][:100], 'o', color='yellow')


    # ax[0].legend(['Кристаллы льда', 'Сухой снег', 'Мокрый снег', 'Дождь', 'Морось'])
    # ax[1].legend(['Кристаллы льда', 'Сухой снег', 'Мокрый снег', 'Дождь', 'Морось'])
    ax[0].set_xlim([-10, 60])
    ax[0].set_ylim([-1, 5])
    ax[0].set_ylabel('Zdr, dB')
    ax[0].grid(True)
    ax[1].set_xlim([-10, 60])
    ax[1].set_ylim([-50, -10])
    ax[1].set_ylabel('LDR, dB')
    ax[3].set_xlabel('Zh, dBZ')
    ax[1].grid(True)
    # plt.show()

    # fig, ax = plt.subplots(2, 1)
    ax[2].plot(data_ic[:, 0][:100], data_ic[:, 3][:100], '+', color='lightblue')
    ax[3].plot(data_ic[:, 0][:100], data_ic[:, 4][:100], '+', color='lightblue')
    ax[2].plot(data_ds[:, 0][:100], data_ds[:, 3][:100], '*', color='green')
    ax[3].plot(data_ds[:, 0][:100], data_ds[:, 4][:100], '*', color='green')

    ax[2].plot(data_ws[:, 0][:100], data_ws[:, 3][:100], '+', color='darkblue')
    ax[3].plot(data_ws[:, 0][:100], data_ws[:, 4][:100], '+', color='darkblue')

    ax[2].plot(data_r[:, 0][:100], data_r[:, 3][:100], 'o', color='red')
    ax[3].plot(data_r[:, 0][:100], data_r[:, 4][:100], 'o', color='red')

    ax[2].plot(data_drizzle[:, 0][:100], data_drizzle[:, 3][:100], 'o', color='yellow')
    ax[3].plot(data_drizzle[:, 0][:100], data_drizzle[:, 4][:100], 'o', color='yellow')

    # ax[2].legend(['Кристаллы льда', 'Сухой снег', 'Мокрый снег', 'Дождь', 'Морось'])
    ax[3].legend(['Кристаллы льда', 'Сухой снег', 'Мокрый снег', 'Дождь', 'Морось'])
    ax[2].set_xlim([-10, 60])
    ax[2].set_ylim([0.975, 1.005])
    ax[2].grid(True)
    ax[3].set_xlim([-10, 60])
    ax[3].set_ylim([0, 15])
    ax[3].grid(True)

    ax[2].set_ylabel('ρhv')
    ax[3].set_ylabel('Kdp, °')
    plt.show()


def calculate_all():
    data_ic = get_for_compare_ice_crystals()
    with open('ice_crystals.bin', 'ab') as fid:
        fid.write(data_ic.tobytes())
        fid.close()

    data_ws = get_for_compare_wet_snow()
    with open('wet_snow.bin', 'ab') as fid:
        fid.write(data_ws.tobytes())
        fid.close()

    data_rain = get_for_compare_rain()
    with open('rain.bin', 'ab') as fid:
        fid.write(data_rain.tobytes())
        fid.close()

    data_ds = get_for_compare_dry_snow()
    with open('dry_snow.bin', 'ab') as fid:
        fid.write(data_ds.tobytes())
        fid.close()

    data_drizzle = get_for_compare_drizzle()
    with open('drizzle.bin', 'ab') as fid:
        fid.write(data_drizzle.tobytes())
        fid.close()


def calculate_mf():
    data_dzl = get_for_membership_drizzle()
    with open('drizzle_mf.bin', 'wb') as fid:
        fid.write(data_dzl.tobytes())
        fid.close()

    data_lr = get_for_membership_light_rain()
    with open('light_rain.bin', 'wb') as fid:
        fid.write(data_lr.tobytes())
        fid.close()

    data_mr = get_for_membership_medium_rain()
    with open('medium_rain.bin', 'wb') as fid:
        fid.write(data_mr.tobytes())
        fid.close()

    data_hr = get_for_membership_heavy_rain()
    with open('heavy_rain.bin', 'wb') as fid:
        fid.write(data_hr.tobytes())
        fid.close()

    data_ld = get_for_membership_large_drops()
    with open('large_drops.bin', 'wb') as fid:
        fid.write(data_ld.tobytes())
        fid.close()

    data_ws = get_for_membership_wet_snow()
    with open('wet_snow.bin', 'ab') as fid:
        fid.write(data_ws.tobytes())
        fid.close()
    #
    data_ds = get_for_membership_dry_snow()
    with open('dry_snow.bin', 'ab') as fid:
        fid.write(data_ds.tobytes())
        fid.close()

    data_ic = get_for_membership_ice_crystals()
    with open('ice_crystals.bin', 'ab') as fid:
        fid.write(data_ic.tobytes())
        fid.close()

def calculate_classifier():
    num_iterations = 501

    accumulated_points = np.zeros([6, 1])
    accumulated_points[0] = 234
    accumulated_points[1] = 340
    accumulated_points[2] = 500
    accumulated_points[3] = 500
    accumulated_points[4] = 500
    accumulated_points[5] = 343

    num_finished = 0

    gen_ice = get_generator_ice(5, 1000)
    gen_water = get_generator_water(5, 1000)

    with open('cw_25.bin', 'ab') as fid_w25, open('ic_25.bin', 'ab') as fid_i25, \
        open('cw_15.bin', 'ab') as fid_w15, open('ic_15.bin', 'ab') as fid_i15, \
        open('cw_5.bin', 'ab') as fid_w5, open('ic_5.bin', 'ab') as fid_i5:
        while num_finished < 6:
            if sum(accumulated_points[0:3]) < num_iterations * 3:
                values_cw = next(gen_water)
            if sum(accumulated_points[3:6]) < num_iterations * 3:
                values_ic = next(gen_ice)
            if accumulated_points[0] < num_iterations:
                if 5 <= values_cw[0] < 15:
                    fid_w5.write(values_cw.tobytes())
                    accumulated_points[0] += 1
                    if accumulated_points[0] == num_iterations:
                        num_finished += 1
                        fid_w5.close()

            if accumulated_points[1] < num_iterations:
                if 15 <= values_cw[0] < 25:
                    fid_w15.write(values_cw.tobytes())
                    accumulated_points[1] += 1
                    if accumulated_points[1] == num_iterations:
                        num_finished += 1
                        fid_w15.close()

            if accumulated_points[2] < num_iterations:
                if 25 <= values_cw[0]:
                    fid_w25.write(values_cw.tobytes())
                    accumulated_points[2] += 1
                    if accumulated_points[2] == num_iterations:
                        num_finished += 1
                        fid_w25.close()

            if accumulated_points[3] < num_iterations:
                if 5 <= values_ic[0] < 15:
                    fid_i5.write(values_ic.tobytes())
                    accumulated_points[3] += 1
                    if accumulated_points[3] == num_iterations:
                        num_finished += 1
                        fid_i5.close()

            if accumulated_points[4] < num_iterations:
                if 15 <= values_ic[0] < 25:
                    fid_i15.write(values_ic.tobytes())
                    accumulated_points[4] += 1
                    if accumulated_points[4] == num_iterations:
                        num_finished += 1
                        fid_i15.close()

            if accumulated_points[5] < num_iterations:
                if 25 <= values_ic[0]:
                    fid_i25.write(values_ic.tobytes())
                    accumulated_points[5] += 1
                    if accumulated_points[5] == num_iterations:
                        num_finished += 1
                        fid_i25.close()

            print(*accumulated_points, num_finished)

def calculate_detect():
    num_iterations = 900

    with open('cooled_water.bin', 'ab') as fid_w,  open('crystals.bin', 'ab') as fid_i:
        gen_ice = get_generator_ice(5, 1000)
        gen_water = get_generator_water(5, 1000)
        for step in range(num_iterations):
            fid_i.write(next(gen_ice).tobytes())
            fid_w.write(next(gen_water).tobytes())
            print(f'Calculated {step} reports.')
        fid_w.close()
        fid_i.close()



def plot_mf():
    num_points_to_plot = 100

    with open('drizzle_mf.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_dzl = np.frombuffer(raw_bytes, dtype=np.float)
        data_dzl = np.array(data_dzl.reshape([int(len(data_dzl) / 5), 5]))
        fid.close()

    with open('light_rain.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_lr = np.frombuffer(raw_bytes, dtype=np.float)
        data_lr = np.array(data_lr.reshape([int(len(data_lr) / 5), 5]))
        fid.close()

    with open('medium_rain.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_mr = np.frombuffer(raw_bytes, dtype=np.float)
        data_mr = np.array(data_mr.reshape([int(len(data_mr) / 5), 5]))
        fid.close()

    with open('heavy_rain.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_hr = np.frombuffer(raw_bytes, dtype=np.float)
        data_hr = np.array(data_hr.reshape([int(len(data_hr) / 5), 5]))
        fid.close()

    with open('large_drops.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_ld = np.frombuffer(raw_bytes, dtype=np.float)
        data_ld = np.array(data_ld.reshape([int(len(data_ld) / 5), 5]))
        fid.close()

    with open('wet_snow.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_ws = np.frombuffer(raw_bytes, dtype=np.float)
        data_ws = np.array(data_ws.reshape([int(len(data_ws) / 5), 5]))
        fid.close()

    with open('dry_snow.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_ds = np.frombuffer(raw_bytes, dtype=np.float)
        data_ds = np.array(data_ds.reshape([int(len(data_ds) / 5), 5]))
        fid.close()

    with open('ice_crystals.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_ic = np.frombuffer(raw_bytes, dtype=np.float)
        data_ic = np.array(data_ic.reshape([int(len(data_ic) / 5), 5]))
        fid.close()

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(data_dzl[0:num_points_to_plot, 0], data_dzl[0:num_points_to_plot, 1], '+', color='lightblue')
    ax[1].plot(data_dzl[0:num_points_to_plot, 0], data_dzl[0:num_points_to_plot, 2], '+', color='lightblue')
    ax[2].plot(data_dzl[0:num_points_to_plot, 0], data_dzl[0:num_points_to_plot, 4], '+', color='lightblue')

    ax[0].plot(data_lr[0:num_points_to_plot, 0], data_lr[0:num_points_to_plot, 1], '*', color='green')
    ax[1].plot(data_lr[0:num_points_to_plot, 0], data_lr[0:num_points_to_plot, 2], '*', color='green')
    ax[2].plot(data_lr[0:num_points_to_plot, 0], data_lr[0:num_points_to_plot, 4], '*', color='green')

    ax[0].plot(data_mr[0:num_points_to_plot, 0], data_mr[0:num_points_to_plot, 1], '+', color='darkblue')
    ax[1].plot(data_mr[0:num_points_to_plot, 0], data_mr[0:num_points_to_plot, 2], '+', color='darkblue')
    ax[2].plot(data_mr[0:num_points_to_plot, 0], data_mr[0:num_points_to_plot, 4], '+', color='darkblue')

    ax[0].plot(data_hr[0:num_points_to_plot, 0], data_hr[0:num_points_to_plot, 1], 'o', color='red')
    ax[1].plot(data_hr[0:num_points_to_plot, 0], data_hr[0:num_points_to_plot, 2], 'o', color='red')
    ax[2].plot(data_hr[0:num_points_to_plot, 0], data_hr[0:num_points_to_plot, 4], 'o', color='red')

    ax[0].plot(data_ld[0:num_points_to_plot, 0], data_ld[0:num_points_to_plot, 1], 'o', color='yellow')
    ax[1].plot(data_ld[0:num_points_to_plot, 0], data_ld[0:num_points_to_plot, 2], 'o', color='yellow')
    ax[2].plot(data_ld[0:num_points_to_plot, 0], data_ld[0:num_points_to_plot, 4], 'o', color='yellow')

    ax[0].plot(data_ws[0:num_points_to_plot, 0], data_ws[0:num_points_to_plot, 1], 'o', color='orange')
    ax[1].plot(data_ws[0:num_points_to_plot, 0], data_ws[0:num_points_to_plot, 2], 'o', color='orange')
    ax[2].plot(data_ws[0:num_points_to_plot, 0], data_ws[0:num_points_to_plot, 4], 'o', color='orange')

    ax[0].plot(data_ds[0:num_points_to_plot, 0], data_ds[0:num_points_to_plot, 1], '*', color='magenta')
    ax[1].plot(data_ds[0:num_points_to_plot, 0], data_ds[0:num_points_to_plot, 2], '*', color='magenta')
    ax[2].plot(data_ds[0:num_points_to_plot, 0], data_ds[0:num_points_to_plot, 4], '*', color='magenta')

    ax[0].plot(data_ic[0:num_points_to_plot, 0], data_ic[0:num_points_to_plot, 1], '+', color='brown')
    ax[1].plot(data_ic[0:num_points_to_plot, 0], data_ic[0:num_points_to_plot, 2], '+', color='brown')
    ax[2].plot(data_ic[0:num_points_to_plot, 0], data_ic[0:num_points_to_plot, 4], '+', color='brown')



    legends_list = ['Морось', 'Слабый дождь', 'Умеренный дождь', 'Ливневый дождь', 'Большие капли',
                    'Мокрый снег', 'Сухой снег', 'Ориентированные кристаллы льда']
    # ax[0].legend(legends_list)
    # ax[1].legend(legends_list)
    ax[2].legend(legends_list, loc='upper left', ncol=1)
    ax[0].set_xlim([-10, 70])
    ax[0].set_ylim([-2, 4])
    ax[0].grid(True)
    ax[1].set_xlim([-10, 70])
    ax[1].set_ylim([-70, -10])
    ax[1].grid(True)
    ax[2].set_xlim([-10, 70])
    ax[2].set_ylim([-3, 50])
    ax[2].grid(True)

    ax[0].set_ylabel('Zdr, dB')
    ax[2].set_xlabel('Zh, dBZ')
    ax[1].set_ylabel('LDR, dB')
    ax[2].set_ylabel('Kdp, °')

    plt.show()



def get_for_compare_drizzle(init_seed: float = None) -> np.array:
    wavelength = 0.11
    min_temp = -15.
    max_temp = 5.
    num_iters = 900

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_drizzle = SpheroidModel('drizzle', wavelength, 'compare', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_drizzle.get_products(next(t_gen), 7 - 4 * random()))
        print('drz', step)
    out_data = np.array(out_data)
    return out_data


def get_for_compare_rain(init_seed: float = None) -> np.array:
    wavelength = 0.11
    min_temp = -5.
    max_temp = 5.
    num_iters = 900

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('rain', wavelength, 'compare', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 14 - 6 * random()))
        print('r', step)
    out_data = np.array(out_data)
    return out_data


def get_for_compare_wet_snow(init_seed: float = None) -> np.array:
    wavelength = 0.11
    min_temp = -15.
    max_temp = 5.
    num_iters = 900

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('wet_snow', wavelength, 'compare', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 14 - 6 * random()))
        print('ws', step)
    out_data = np.array(out_data)
    return out_data


def get_for_compare_dry_snow(init_seed: float = None) -> np.array:
    wavelength = 0.11
    min_temp = -15.
    max_temp = 0.
    num_iters = 900

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('dry_snow', wavelength, 'compare', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 14 - 6 * random()))
        print('ds', step)
    out_data = np.array(out_data)
    return out_data


def get_for_compare_ice_crystals(init_seed: float = None) -> np.array:
    wavelength = 0.11
    min_temp = -15.
    max_temp = 0.
    num_iters = 900

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('ice_crystals', wavelength, 'compare', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 6 - 2 * random()))
        print('ic', step)
    out_data = np.array(out_data)
    return out_data


def get_for_membership_drizzle(init_seed: float = None) -> np.array:
    wavelength = 0.0325
    min_temp = -15.
    max_temp = 0.
    num_iters = 1000

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_drizzle = SpheroidModel('drizzle', wavelength, 'general', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_drizzle.get_products(next(t_gen), 20 - 10 * random()))
        print('drz', step)
    out_data = np.array(out_data)
    return out_data


def get_for_membership_light_rain(init_seed: float = None) -> np.array:
    wavelength = 0.0325
    min_temp = -5.
    max_temp = 5.
    num_iters = 1000

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('light_rain', wavelength, 'general', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 14 - 6 * random()))
        print('lr', step)
    out_data = np.array(out_data)
    return out_data

def get_for_membership_medium_rain(init_seed: float = None) -> np.array:
    wavelength = 0.0325
    min_temp = -5.
    max_temp = 5.
    num_iters = 1000

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('medium_rain', wavelength, 'general', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 14 - 6 * random()))
        print('mr', step)
    out_data = np.array(out_data)
    return out_data

def get_for_membership_heavy_rain(init_seed: float = None) -> np.array:
    wavelength = 0.0325
    min_temp = -5.
    max_temp = 5.
    num_iters = 1000

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('heavy_rain', wavelength, 'general', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 14 - 6 * random()))
        print('r', step)
    out_data = np.array(out_data)
    return out_data

def get_for_membership_large_drops(init_seed: float = None) -> np.array:
    wavelength = 0.0325
    min_temp = -5.
    max_temp = 5.
    num_iters = 1000

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('large_drops', wavelength, 'general', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 10 - 6 * random()))
        print('ld', step)
    out_data = np.array(out_data)
    return out_data


def get_for_membership_wet_snow(init_seed: float = None) -> np.array:
    wavelength = 0.0325
    min_temp = -15.
    max_temp = 5.
    num_iters = 900

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('wet_snow', wavelength, 'general', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 14 - 6 * random()))
        print('ws', step)
    out_data = np.array(out_data)
    return out_data


def get_for_membership_dry_snow(init_seed: float = None) -> np.array:
    wavelength = 0.0325
    min_temp = -15.
    max_temp = 0.
    num_iters = 900

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('dry_snow', wavelength, 'general', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 14 - 6 * random()))
        print('ds', step)
    out_data = np.array(out_data)
    return out_data


def get_for_membership_ice_crystals(init_seed: float = None) -> np.array:
    wavelength = 0.0325
    min_temp = -15.
    max_temp = 0.
    num_iters = 900

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    model_rain = SpheroidModel('ice_crystals', wavelength, 'general', init_seed)
    out_data = []
    for step in range(num_iters):
        out_data.append(model_rain.get_products(next(t_gen), 14 - 6 * random()))
        print('ic', step)
    out_data = np.array(out_data)
    return out_data


def get_generator_ice(min_dbz, max_dbz, init_seed: float = None):
    wavelength = 0.0325
    min_temp = -15.
    max_temp = 0.

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    class_gen = UniformRandomValue(0., 1.999999999999999999, init_seed)
    models = [SpheroidModel('ice_crystals', wavelength, 'general', init_seed),
              SpheroidModel('dry_snow', wavelength, 'general', init_seed)]

    def generator_ice():
        while True:
            class_out = int(next(class_gen))
            while True:
                Zh, Zdr, LDR, ro_hv, Kdp = models[class_out].get_products(next(t_gen), 14 - 6 * random())
                if min_dbz <= Zh <= max_dbz:
                    break
            yield np.array([Zh, Zdr, LDR, Kdp], dtype=np.float64)

    return generator_ice()

def get_generator_water(min_dbz, max_dbz, init_seed: float = None):

    wavelength = 0.0325
    min_temp = -15.
    max_temp = 0.

    t_gen = UniformRandomValue(min_temp, max_temp, init_seed)
    class_gen = UniformRandomValue(0., 5.999999999999999999, init_seed)
    models = [SpheroidModel('drizzle', wavelength, 'general', init_seed),
              SpheroidModel('light_rain', wavelength, 'general', init_seed),
              SpheroidModel('medium_rain', wavelength, 'general', init_seed),
              SpheroidModel('heavy_rain', wavelength, 'general', init_seed),
              SpheroidModel('wet_snow', wavelength, 'general', init_seed),
              SpheroidModel('large_drops', wavelength, 'general', init_seed)]

    fa = [(20, 10), (14, 6),  (14, 6) , (14, 6), (14, 6), (10, 6)]

    def generator_water():
        while True:
            class_out = int(next(class_gen))
            while True:
                Zh, Zdr, LDR, ro_hv, Kdp = models[class_out].get_products(next(t_gen), fa[class_out][0] - fa[class_out][1] * random())
                if min_dbz <= Zh <= max_dbz:
                    break
            # print('returning water', Zh, Zdr)
            yield np.array([Zh, Zdr, LDR, Kdp], dtype=np.float64)

    return generator_water()

def main():
   # path_to_data = 'E:/Git/weather-radar/weather-radar-math/bin'
    path_to_data = '/media/serj/Data/Git/repos/weather-radar-math/bin_classify'


    # path_to_data = '/media/serj/Data/Git/repos/weather-radar-math/bin_final_2000'
    if not os.path.exists(path_to_data):
        os.mkdir(path_to_data)

    os.chdir(path_to_data)

    # calculate_all()
    # calculate_mf()
    # calculate_detect()
    calculate_classifier()
    # plot_all()
    # plot_mf()
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