#!/usr/bin/env python3
from random import seed, gauss, uniform
from typing import Generator

def GaussRandomValue(mean_val: float, dispersion: float, initializer=None) -> Generator:
    """
    Генератор случайной величины, распределенной по гауссому закону

    :param mean_val:
    :param dispersion:
    :param initializer:
    :return:
    """
    seed(initializer)
    while True:
        yield gauss(mean_val, dispersion)

def UniformRandomValue(min_val: float, max_val: float, initializer=None) -> Generator:
    """
    Генератор случайной величины, распределенной по равномерному случайному закону

    :param min_val:
    :param max_val:
    :param initializer:
    :return:
    """
    seed(initializer)
    while True:
        yield uniform(min_val, max_val)

def exponent_size_distribution(_concentration: int, _slope: float):
    def exp_distr_expression(_diameter: float):
        return _concentration * exo
    return exp_distr_expression