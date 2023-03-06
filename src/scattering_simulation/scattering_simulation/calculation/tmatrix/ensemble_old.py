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
