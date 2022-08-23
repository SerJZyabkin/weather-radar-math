from .ampld_par import *
import numpy as np
from dataclasses import dataclass
from ..common_expressions import UniformRandomValue, GaussRandomValue
from random import seed
from abc import ABC, abstractmethod
from types import GeneratorType, FunctionType
from typing import Union
from ..particle_properties.rain.drop_size import

_INPUT_MASK = {'min_diameter': 1, 'max_diameter': 2, 'mean_diameter': 4, }

class SpheroidInfoFactory(ABC):
    @abstractmethod
    def get_diameter_model(self) -> (FunctionType, int):
        pass

    @abstractmethod
    def get_dimensions_model(self) -> (FunctionType, int):
        pass

    @abstractmethod
    def get_tilt_model(self) -> (FunctionType, int):
        pass

    @abstractmethod
    def get_permittivity_model(self) -> (FunctionType, int):
        pass

class RainDropletsInfoFactory(SpheroidInfoFactory):
    def get_diameter_model(self) -> GeneratorType:
        return 1

    @abstractmethod
    def get_dimensions_model(self) -> FunctionType:
        return 4


    @abstractmethod
    def get_tilt_model(self) -> GeneratorType:
        return 3


    @abstractmethod
    def get_permittivity_model(self) -> FunctionType:
        return 2


def get_spheroid_info(class_type: str) -> SpheroidInfoFactory:
    factories = {'rain_droplets': RainDropletsInfoFactory(),}
    if class_type in factories.keys()
        return factories[class_type]



@dataclass
class SpheroidModelSettings:
    """Настройки для генерации поляриметрических продуктов метеолокатов"""
    name_diameter_model: str
    name_dimensions_model: str
    name_tilt_model: str
    permittivity_model: str

def generator_temperature(temperature: (float, float)):
    generator = UniformRandomValue(*temperature)
    while True:
        yield next(generator)

class SpheroidModel:

    def __init__(self, settings: SpheroidModelSettings, temperature: (float, float), wavelength: float,
                 init_seed: float = None):
        self.model_temperature = UniformRandomValue(*temperature, init_seed)


    def get(self):
        print(next(self.model_temperature))
        # calculate(wavelength, Deq, epsilon, gamma, alpha_tilt, beta_tilt, theta)

def main():
    settings = SpheroidModelSettings('rain_droplets_diameter', 'rain_droplets_dimensions', 'rain_droplets_tilt',
                                     'rain_droplets_permittivity')
    model = SpheroidModel(settings, (0, 10), 0.0325, 10)
    model.get()