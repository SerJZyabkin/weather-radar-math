from .tmatrix.ampld_lp import calculate
import inspect
from .common_expressions import GaussRandomValue, UniformRandomValue
from typing import Generator
from .particle_properties.rain.drop_shape import get_gamma_rain
from .particle_properties.rain.drop_size import factory_normalized_gamma_distribution, \
    factory_normalized_exponential_distribution
from .particle_properties.dielectric_permittivity.lossy_medium import get_water_reflection_index, \
    get_air_dielectric_constant, get_ice_dielectric_constant
from scipy.integrate import quad
from numpy import pi
import numpy as np
from random import random, seed
from .particle_properties.dielectric_permittivity.medium_mixture import get_mixing_ponder_vansandern
from .particle_properties.dielectric_permittivity.particular_mixtures import get_mixing_for_wet_snow

def water_diel_permit_model(wave_length:float, min_temperature: float, max_temperature: float) -> np.complex64:
    gen_temperature = UniformRandomValue(min_temperature, max_temperature)
    while True:
        cur_temp = next(gen_temperature)
        yield get_water_reflection_index(wave_length, cur_temp)


def dry_snowflake_diel_permit_model(wave_length:float, min_temperature: float, max_temperature: float,
                                min_air_incursion: float, max_air_incursion: float) -> np.complex64:
    gen_temperature = UniformRandomValue(min_temperature, max_temperature)
    if min_air_incursion >= max_air_incursion or max_air_incursion >= 1.:
        print(f'dry_snowflake_diel_permit_model: Bad incusion {min_air_incursion} / {max_air_incursion}')
        quit()

    gen_incursion = UniformRandomValue(min_air_incursion, max_air_incursion)
    while True:
        cur_temp = next(gen_temperature)
        cur_incursion = next(gen_incursion)
        e1 = get_ice_dielectric_constant(wave_length, cur_temp)
        e2 = get_air_dielectric_constant(wave_length)
        yield np.sqrt(get_mixing_ponder_vansandern(e1, e2, 1 - cur_incursion, cur_incursion))



def wet_snowflake_diel_permit_model(wave_length:float, min_temperature: float, max_temperature: float, min_melt: float,
                                    max_melt: float, min_density: float, max_density: float) -> np.complex64:
    gen_temperature = UniformRandomValue(min_temperature, max_temperature)
    gen_melt = UniformRandomValue(min_melt, max_melt)
    gen_density = UniformRandomValue(min_density, max_density)
    while True:
        cur_temp = next(gen_temperature)
        cur_melt = next(gen_melt)
        cur_density = next(gen_density)
        yield np.sqrt(get_mixing_for_wet_snow(wave_length, cur_temp, cur_density, cur_melt))

class ParticleStatsModel:
    def __init__(self, _gen_mean_diameter: Generator, _gen_orientation: Generator, _diel_permit_gen: Generator):
        self.gen_mean_diameter = _gen_mean_diameter
        self.gen_orientation = _gen_orientation
        self._diel_permit_gen = _diel_permit_gen

    def get_single_orientation(self):
        return next(self.gen_orientation)

    def get_dielectric_permittivity(self) -> float:
        return next(self._diel_permit_gen)

    def get_single_size_model(self) -> (int, 'function'):
        raise NotImplementedError(f'\nMethod {inspect.stack()[0][3]} is missing for {type(self).__name__} class.')

    def get_single_shape_model(self) -> (int, 'function'):
        raise NotImplementedError(f'\nMethod {inspect.stack()[0][3]} is missing for {type(self).__name__} class.')


class WaterDropStatsModel(ParticleStatsModel):
    def __init__(self, _diameter_min: float, _diameter_max: float, _tilt_mean: float, _tilt_dispersion: float,
                 _con_param_min: int, _con_param_max: int, _shape_min: float, _shape_max: float,
                 _diel_permit_gen: Generator):
        diameter_model = UniformRandomValue(_diameter_min, _diameter_max)
        tilt_model = GaussRandomValue(_tilt_mean, _tilt_dispersion)
        super(WaterDropStatsModel, self).__init__(diameter_model, tilt_model, _diel_permit_gen)

        self.shape_model = UniformRandomValue(_shape_min, _shape_max)
        self.concentration_param_model = UniformRandomValue(_con_param_min, _con_param_max)

    def get_single_size_model(self) -> 'function':
        func_out = factory_normalized_gamma_distribution(next(self.concentration_param_model), next(self.shape_model),
                                                          next(self.gen_mean_diameter))
        return func_out

    def get_single_shape_model(self) -> 'function':
        return get_gamma_rain


class SnowflakeStatsModel(ParticleStatsModel):
    def __init__(self, _diameter_min: float, _diameter_max: float, _tilt_mean: float, _tilt_dispersion: float,
                 _n_param_min: int, _n_param_max: int, _min_slope: float, _max_slope: float, _min_gamma: float,
                 _max_gamma: float, _diel_permit_gen: Generator):

        diameter_model = UniformRandomValue(_diameter_min, _diameter_max)
        tilt_model = GaussRandomValue(_tilt_mean, _tilt_dispersion)

        super(SnowflakeStatsModel, self).__init__(diameter_model, tilt_model, _diel_permit_gen)

        self.min_gamma = _min_gamma
        self.max_gamma = _max_gamma
        self.slope_model = UniformRandomValue(_min_slope, _max_slope)
        self.n_param_model = UniformRandomValue(_n_param_min, _n_param_max)

    def get_single_size_model(self) -> 'function':
        func_out = factory_normalized_exponential_distribution(next(self.n_param_model), next(self.slope_model),
                                                               next(self.gen_mean_diameter))
        return func_out

    def get_single_shape_model(self) -> 'function':
        added_gamma = 0.1 * random()
        gen_gamma_value = UniformRandomValue(self.min_gamma, self.max_gamma)

        def get_gamma_snow(particle_size: float):
            return added_gamma + next(gen_gamma_value)

        return get_gamma_snow

class CalculatorProducts:
    def __init__(self, particle_stats: ParticleStatsModel):
        self.psm = particle_stats

    def calculate(self,min_diam: float, diam_delta:float, diam_numsteps: int) -> (float, float, float):
        z_h = 0
        z_v = 0
        z_hv = 0
        z_dr = 0
        ldr = 0
        kdp = 0

        func_size = self.psm.get_single_size_model()
        func_size_inc =  UniformRandomValue(0, diam_delta)
        func_shape = self.psm.get_single_shape_model()

        particle_info = list()

        cur_min_diam = 0
        cur_max_diam = diam_delta
        for _ in range(diam_numsteps):
            particle_info.append((int(np.round(quad(func_size, cur_min_diam, cur_max_diam)[0])), cur_min_diam))
            cur_min_diam = cur_max_diam
            cur_max_diam += diam_delta
        nn = 0
        for cur_pi in particle_info:
            if cur_pi[0] < 1:
                continue
            nn += cur_pi[0]
            particle_size = cur_pi[1] + next(func_size_inc)
            particle_gamma = func_shape(particle_size)
            tilt_alpha = self.psm.get_single_orientation()
            tilt_beta = self.psm.get_single_orientation()

            s11, s12, s21, s22 = calculate(32.5, particle_size, self.psm.get_dielectric_permittivity(),
                                           particle_gamma, tilt_alpha, 90. + tilt_beta, 0)
            # s11f, _, _, s22f = calculate(32.5, particle_size, self.psm.get_dielectric_permittivity(),
            #                                particle_gamma, tilt_alpha, 90. + tilt_beta, 180)

            z_h += cur_pi[0] * np.abs(s11)**2
            z_v += cur_pi[0] * np.abs(s22)**2
            z_hv += cur_pi[0] * np.abs(s21) ** 2
            kdp += cur_pi[0] * np.real(s11 - s22)
           # print(f'Finished step {z_h}, {z_v}, {10* np.log10(z_h / z_v)}')

        ZH = z_h * 4 * 32.5**4 / pi**4/ 0.9 * 1e-3
        ZV = z_v * 4 * 32.5**4 / pi ** 4 / 0.9 * 1e-3
        ZVH = z_hv * 4 * 32.5 ** 4 / pi ** 4 / 0.9 * 1e-3
        print(f'Zh = {ZH} = {10 * np.log10(ZH)}')
        print(f'Zv = {ZV} = {10 * np.log10(ZV)}')
        print(f'Zdr = {10 * (np.log10(ZH) - np.log10(ZV))}')
        print(f'LDR = {10 * (np.log10(ZVH) - np.log10(ZH))}')
        print(f'Kdp = {kdp * 0.18 / pi * 0.0325}')
        return z_h, z_dr, ldr