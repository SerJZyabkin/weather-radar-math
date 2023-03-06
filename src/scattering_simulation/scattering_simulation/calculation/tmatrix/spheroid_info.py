import random

import numpy as np
from sys import exit
from ..common_expressions import UniformRandomValue, GaussRandomValue
from abc import ABC, abstractmethod
from typing import Callable, Iterator
from ..particle_properties.rain.drop_size import factory_exponential_distribution, \
    factory_normalized_gamma_distribution, factory_normalized_exponential_distribution
from ..particle_properties.rain.drop_shape import get_gamma_rain
from ..particle_properties.dielectric_permittivity.lossy_medium import get_ice_reflection_index, \
    get_water_reflection_index, get_water_dielectric_constant, get_ice_dielectric_constant, get_air_dielectric_constant
from ..particle_properties.dielectric_permittivity.particular_mixtures import get_mixing_for_wet_snow
from ..particle_properties.dielectric_permittivity.medium_mixture import get_mixing_ponder_vansandern
from random import uniform


def generator_const(value):
    while True:
        yield value

class SpheroidInfoFactory(ABC):
    def __init__(self, min_diameter: float, max_diameter: float, init_seed: float = None):
        self.init_seed = init_seed
        self.min_diameter = min_diameter
        self.max_diameter = max_diameter
        random.seed(init_seed)

    def get_min_diameter(self) -> float:
        return self.min_diameter

    def get_max_diameter(self) -> float:
        return self.max_diameter

    @staticmethod
    @abstractmethod
    def get_distribution_model() -> (Callable[[float], int], int):
        pass

    @staticmethod
    @abstractmethod
    def get_dimensions_model() -> Callable[[float], float]:
        pass

    @abstractmethod
    def get_tilt_model(self) -> Iterator[float]:
        pass

    @abstractmethod
    def get_permittivity_model(self) -> Callable[[float, float], complex]:
        """Комплексный показатель преломления """
        pass


def get_rain_class(subclass_name: str):
    # Основные классы
    if subclass_name == 'drizzle':
        num_bins_distribution = 20
        min_diameter = 0.6
        max_diameter = 7
        model_number_concentration = UniformRandomValue(1e3, 2.1e4)
        model_shape = UniformRandomValue(-1, 4)
        model_norm_diameter = UniformRandomValue(0.5, 1.4)
    elif subclass_name == 'medium_rain':
        num_bins_distribution = 20
        min_diameter = 0.6
        max_diameter = 7
        model_number_concentration = UniformRandomValue(1e3, 1e4)
        model_shape = UniformRandomValue(-1, 4)
        model_norm_diameter = UniformRandomValue(1.4, 2)
    elif subclass_name == 'heavy_rain':
        num_bins_distribution = 20
        min_diameter = 0.6
        max_diameter = 7
        model_number_concentration = UniformRandomValue(2e3, 9e3)
        model_shape = UniformRandomValue(-1, 4)
        model_norm_diameter = UniformRandomValue(1.8, 3.2)
    elif subclass_name == 'large_drops':
        num_bins_distribution = 20
        min_diameter = 0.6
        max_diameter = 7
        model_number_concentration = UniformRandomValue(15, 150)
        model_shape = UniformRandomValue(-0.94, 0.87)
        model_norm_diameter = UniformRandomValue(1.3, 3.6)

    # Класс для рисования рисунков в пояснительную записку
    elif subclass_name == 'rain_fixed':
        num_bins_distribution = 20
        min_diameter = 0.5
        max_diameter = 7
        model_number_concentration = generator_const(1e3)
        model_shape = generator_const(-0.9)
        model_norm_diameter = generator_const(1)

    # Классы для сравнения с "эталонными" данными из статьи "Supervised Fuzzy-Logic Classification of Hydrometeor
    # using C-band Weather Radar
    elif subclass_name == 'drizzle_compare':
        num_bins_distribution = 30
        min_diameter = 0.6
        max_diameter = 3
        angle_dispersion = 10
        min_mean_angle = 5
        max_mean_angle = 10

        def rain_tilt_model(mean_angle) -> Iterator[float]:
            return GaussRandomValue(mean_angle, angle_dispersion)

        model_number_concentration = UniformRandomValue(1e2, 8e2)
        model_shape = UniformRandomValue(-1, 1)
        model_norm_diameter = UniformRandomValue(0.4, 1.6)

    elif subclass_name == 'rain_compare':
        num_bins_distribution = 30
        min_diameter = 0.6
        max_diameter = 7
        angle_dispersion = 2.5
        min_mean_angle = 9
        max_mean_angle = 9

        def rain_tilt_model(mean_angle) -> Iterator[float]:
            return GaussRandomValue(mean_angle, angle_dispersion)

        model_number_concentration = UniformRandomValue(2e2, 1e4)
        model_shape = UniformRandomValue(-1, 4)
        model_norm_diameter = UniformRandomValue(1.0, 2.8)

    else:
        exit(f'spheroid_info.get_rain_class: Unknown rain subclass name [ {subclass_name} ] received. Exiting.')

    class RainDropletsInfoFactory(SpheroidInfoFactory):
        def __init__(self, init_seed: float):
            super().__init__(min_diameter, max_diameter, init_seed)

        @staticmethod
        def get_distribution_model() -> (Callable[[float], float], int):
            out_func = factory_normalized_gamma_distribution(next(model_number_concentration), next(model_shape),
                                                             next(model_norm_diameter))
            return out_func, num_bins_distribution

        @staticmethod
        def get_dimensions_model() -> Callable[[float], float]:
            return get_gamma_rain

        def get_tilt_model(self) -> Iterator[float]:
            mean_angle = uniform(min_mean_angle, max_mean_angle)
            return rain_tilt_model(mean_angle)

        def get_permittivity_model(self) -> Callable[[float, float], complex]:
            return get_water_reflection_index

    return RainDropletsInfoFactory


def get_dry_snow_class(subclass_name: str):
    if subclass_name == 'dry_snow':
        num_bins_distribution = 14
        min_diameter = 1
        max_diameter = 15
        model_slope = UniformRandomValue(2.5, 8.2)
        model_intercept = UniformRandomValue(2380, 42000)
        mixing_percent = generator_const(0.2)

    elif subclass_name == 'dry_snow_fixed':
        num_bins_distribution = 14
        min_diameter = 1
        max_diameter = 15

        def dry_snow_tilt_model(mean_angle) -> Iterator[float]:
            generator_const(0.)

        def dry_snow_dimensions_model(equdiam: float):
            return 0.8

        model_slope = generator_const(2.2)
        model_intercept = generator_const(4.2e4)
        mixing_percent = generator_const(0.2)

    elif subclass_name == 'dry_snow_compare':
        num_bins_distribution = 14
        min_diameter = 1
        max_diameter = 15
        angle_dispersion = 5
        min_mean_angle = 5
        max_mean_angle = 5

        def dry_snow_tilt_model(mean_angle) -> Iterator[float]:
            return GaussRandomValue(mean_angle, angle_dispersion)

        def dry_snow_dimensions_model(equdiam: float):
            return uniform(0.8, 1.1)

        model_slope = UniformRandomValue(2.5, 8.2)
        model_intercept = UniformRandomValue(2380, 40000)
        mixing_percent = UniformRandomValue(0.05, 0.2)

    class DrySnowInfoFactory(SpheroidInfoFactory):
        def __init__(self, init_seed: float):
            super().__init__(min_diameter, max_diameter, init_seed)

        def get_distribution_model(self) -> (Callable[[float], float], int):

            out_func = factory_exponential_distribution(next(model_intercept), next(model_slope))
            return out_func, num_bins_distribution

        @staticmethod
        def get_dimensions_model() -> Callable[[float], float]:
            return dry_snow_dimensions_model

        def get_tilt_model(self) -> Iterator[float]:
            mean_angle = uniform(min_mean_angle, max_mean_angle)
            return dry_snow_tilt_model(mean_angle)

        def get_permittivity_model(self) -> Callable[[float, float], complex]:
            def dry_ice_permittivity_model(wavelength: float, temperature: float):
                cur_mixing_percent = next(mixing_percent)
                epsi_1 = get_air_dielectric_constant(wavelength)
                epsi_2 = get_ice_dielectric_constant(wavelength, temperature)
                return np.sqrt(get_mixing_ponder_vansandern(epsi_1, epsi_2, cur_mixing_percent, 1 - cur_mixing_percent))

            return dry_ice_permittivity_model
    return DrySnowInfoFactory


def get_wet_snow_class(subclass_name: str):
    if subclass_name == 'wet_snow':
        num_bins_distribution = 14
        min_diameter = 1
        max_diameter = 15
        angle_dispersion = 5
        min_mean_angle = 10
        max_mean_angle = 80

        def wet_snow_tilt_model(mean_angle) -> Iterator[float]:
            return GaussRandomValue(mean_angle, angle_dispersion)

        def wet_snow_dimensions_model(equdiam: float):
            return uniform(0.7, 0.9)

        model_slope = UniformRandomValue(1.8, 3.1)
        model_intercept = UniformRandomValue(1500, 4800)
        mixing_percent = generator_const(0.2)

    elif subclass_name == 'wet_snow_compare':
        num_bins_distribution = 20
        min_diameter = 1
        max_diameter = 15
        angle_dispersion = 10   # 5 - 18 - 32
        min_mean_angle = 13
        max_mean_angle = 29

        def wet_snow_tilt_model(mean_angle) -> Iterator[float]:
            return GaussRandomValue(mean_angle, angle_dispersion)

        def wet_snow_dimensions_model(equdiam: float):
            return uniform(0.6, 0.7)

        model_slope = UniformRandomValue(2, 8.1)
        model_intercept = UniformRandomValue(1500, 4800)
        mixing_percent = UniformRandomValue(0.1, 0.3)

    elif subclass_name == 'wet_snow_fixed':
        num_bins_distribution = 14
        min_diameter = 1
        max_diameter = 15

        def wet_snow_tilt_model(mean_angle) -> Iterator[float]:
            return generator_const(0.)

        def wet_snow_dimensions_model(equdiam: float):
            return 0.8

        model_slope = generator_const(1.8)
        model_intercept = generator_const(1550)
        mixing_percent = generator_const(0.2)

    class WetSnowInfoFactory(SpheroidInfoFactory):
        def __init__(self, init_seed: float):
            # TODO
            super().__init__(min_diameter, max_diameter, init_seed)

        def get_distribution_model(self) -> (Callable[[float], float], int):
            out_func = factory_exponential_distribution(next(model_intercept), next(model_slope))
            return out_func, num_bins_distribution

        @staticmethod
        def get_dimensions_model() -> Callable[[float], float]:
            return wet_snow_dimensions_model

        def get_tilt_model(self) -> Iterator[float]:
            # TODO
            mean_angle = uniform(min_mean_angle, max_mean_angle)
            return wet_snow_tilt_model(mean_angle)

        def get_permittivity_model(self) -> Callable[[float, float], complex]:
            def wet_ice_permittivity_model(wavelength: float, temperature: float):
                cur_mixing_percent = next(mixing_percent)
                epsi_1 = get_water_dielectric_constant(wavelength, temperature)
                epsi_2 = get_ice_dielectric_constant(wavelength, temperature)
                return np.sqrt(get_mixing_ponder_vansandern(epsi_1, epsi_2, cur_mixing_percent, 1 - cur_mixing_percent))

            return wet_ice_permittivity_model

    return WetSnowInfoFactory


def get_ice_crystal_class(subclass_name: str):

    if subclass_name == 'ice_crystals':
        num_bins_distribution = 14
        min_diameter = 0.2
        max_diameter = 4.8
        min_mean_angle = 10
        max_mean_angle = 80

        def ice_crystals_tilt_model(mean_angle) -> Iterator[float]:
            return GaussRandomValue(mean_angle, angle_dispersion)

        def ice_crystals_dimensions_model(equdiam: float) -> float:
            return uniform(0.7, 0.9)

        model_slope = UniformRandomValue(1.1, 3.0)
        model_intercept = UniformRandomValue(1., 41)

    if subclass_name == 'ice_crystals_compare':

        num_bins_distribution = 20
        min_diameter = 0.2
        max_diameter = 10.8
        angle_dispersion = 5
        min_mean_angle = 4
        max_mean_angle = 12

        def ice_crystals_tilt_model(mean_angle) -> Iterator[float]:
            return GaussRandomValue(mean_angle, angle_dispersion)

        def ice_crystals_dimensions_model(equdiam: float) -> float:
            if equdiam < 0.0015:
                return uniform(0.35, 1)
            else:
                return uniform(0.2, 0.9)

        model_slope = UniformRandomValue(1.1, 3.0)
        model_intercept = UniformRandomValue(60., 120)

    class IceCrystalsInfoFactory(SpheroidInfoFactory):
        def __init__(self, init_seed: float):
            # TODO
            random.seed(init_seed)
            super().__init__(min_diameter, max_diameter, init_seed)

        def get_distribution_model(self) -> (Callable[[float], float], float):
            # TODO
            out_func = factory_exponential_distribution(next(model_intercept), next(model_slope))
            return out_func, num_bins_distribution

        @staticmethod
        def get_dimensions_model() -> Callable[[float], float]:
            # TODO
            return ice_crystals_dimensions_model

        def get_tilt_model(self) -> Iterator[float]:
            # TODO
            mean_angle = uniform(min_mean_angle, max_mean_angle)
            return ice_crystals_tilt_model(mean_angle)

        def get_permittivity_model(self) -> Callable[[float, float], complex]:
            return get_ice_reflection_index

    return IceCrystalsInfoFactory


def get_spheroid_info(class_type: str, init_seed: float) -> SpheroidInfoFactory:
    factories = {'drizzle': get_rain_class('drizzle'),
                 'medium_rain': get_rain_class('medium_rain'),
                 'heavy_rain': get_rain_class('heavy_rain'),
                 'large_drops': get_rain_class('large_drops'),
                 'dry_snow': get_dry_snow_class('dry_snow'),
                 'wet_snow': get_wet_snow_class('wet_snow'),
                 'ice_crystals': get_ice_crystal_class('ice_crystals'),
                 }
    if class_type in factories.keys():
        return factories[class_type](init_seed)


def get_fixed_spheroid_info(class_type: str, init_seed: float) -> SpheroidInfoFactory:
    factories = {'rain': get_rain_class('rain_fixed'),
                 'dry_snow': get_dry_snow_class('dry_snow_fixed'),
                 'wet_snow': get_wet_snow_class('wet_snow_fixed'),
                 }
    if class_type in factories.keys():
        return factories[class_type](init_seed)


def get_spheroid_comparation(class_type: str, init_seed: float) -> SpheroidInfoFactory:
    factories = {'drizzle': get_rain_class('drizzle_compare'),
                 'rain': get_rain_class('rain_compare'),
                 'dry_snow': get_dry_snow_class('dry_snow_compare'),
                 'wet_snow': get_wet_snow_class('wet_snow_compare'),
                 'ice_crystals': get_ice_crystal_class('ice_crystals_compare'),
                 }
    if class_type in factories.keys():
        return factories[class_type](init_seed)


