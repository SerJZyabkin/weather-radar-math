#!/usr/bin/env python3
from numpy import pi
from ..constants import LIGHTSPEED

def get_wave_number(wave_length: float) -> float:
    """
    Процедура расчета волнового числа к

    :param wave_length: длина волны сигнала в метрах
    :return: волновое число к размерности радиан / метр
    """
    return 2. * pi / wave_length


def convert_to_wave_length(frequency: float) -> float:
    """
    Процедура перевода частоты сигнала в длину волны

    :param frequency: частота несущей сигнала в Гц
    :return: длина волны сигнала в метрах
    """
    return LIGHTSPEED / frequency


def convert_to_frequency(wave_length: float) -> float:
    """
    Процедура перевода длины волны несущей сигнала в частоту

    :param wave_length: длина волны несущей сигнала в метрах
    :return: частота сигнала в Гц
    """
    return LIGHTSPEED / wave_length