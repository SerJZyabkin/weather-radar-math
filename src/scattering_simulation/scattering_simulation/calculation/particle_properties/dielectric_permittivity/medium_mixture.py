#!/usr/bin/env python3
"""
`scattering_simulation.calculation.dielectric_permittivity.medium_mixture`
"""
import numpy as np

def get_mixing_maxwell_garnett(epsi_background: complex, epsi_incursion: complex, f_background: float,
                               f_incursion: float) -> complex:
    """
    Расчет диэлектрической проницаемости смеси двух веществ по формуле Максвелла-Гарнетта. Формула высчетана из
    предположения, вклад смешиваемого вещества в результирующее значение значительно меньше, чем фонового, т.е.
    f * y << 1, где y - переменная, рассчитываемся в коде. f - доля смешиваемого вещества.

    :param epsi_background: диэлектрическая проницаемость "фонового" (основного) вещества
    :param epsi_incursion: диэлектрическая проницаемость "вторгающегося" (смешиваемого) вещества
    :param f_background: доля проницаемость "фонового" (основного) вещества
    :param f_incursion: доля "вторгающегося" (смешиваемого) вещества
    :return:
    """
    # Нормализуем доли компонент вещества, чтобы они давали единицу
    _f = f_incursion / (f_background + f_incursion)
    # Расчет выходного значения
    _y = (epsi_incursion - epsi_background) / (epsi_incursion + 2 * epsi_background)
    return epsi_background * (1 + 2 * _f * _y) / (1 - f_incursion * _y)


def get_mixing_ponder_vansandern(epsi_1: complex, epsi_2: complex, f_1: float, f_2: float) -> complex:
    """
    Расчет диэлектрической проницаемости смеси двух веществ по формуле Пондера-ВанСандерна. Результат находится при
    помощи решения квадратичного уравнения. Выбирается корень, у которого действительная часть положительна.

    :param epsi_1: диэлектрическая проницаемость первого смешиваемого вещества
    :param epsi_2: диэлектрическая проницаемость второго смешиваемого вещества
    :param f_1: доля объема первого смешиваемого вещества
    :param f_2: доля объема второго смешиваемого вещества
    :return:
    """
    # Нормализуем доли компонент вещества, чтобы они давали единицу
    _f1n = f_1 / (f_1 + f_2)
    _f2n = 1 - f_1
    # Коэффициенты квадратичного уравнения
    _a = -2
    _b = (2 * epsi_1 - epsi_2) * _f1n + (2 * epsi_2 - epsi_1) * _f2n
    _c = epsi_1 * epsi_2
    # Поиск корней квардатичного уравнения
    _quadratic_roots = np.roots([_a, _b, _c])
    if sum(_quadratic_roots.real > 0) == 1:  # Один из корней уравнения должен быть положительным
        for _root in _quadratic_roots:  # Мы его и ищем и возвращаем
            if _root.real > 0:
                return _root
    # Иначе была обнаружена какая-то проблема, скорее всего некорректный ввод данных!
    quit("Ponder-Vansanders mixing: Root solve error 1 detected.")


def get_mixing_three_components(epsi_1: complex, epsi_2: complex, epsi_3: complex, f_1: float,
                                f_2: float, f_3: float) -> complex:
    # Нормализуем выходные данные
    sum_f = f_1 + f_2 + f_3
    _f1n = f_1 / sum_f
    _f2n = f_2 / sum_f
    _f3n = f_3 / sum_f
    # Расчет диэлектрической постоянной для смеси первых двух компонент по формуле Пондера-ван-Сандерна
    epsi_12 = get_mixing_ponder_vansandern(epsi_1, epsi_2, _f1n, _f2n)
    # Расчет диэлектрической постоянной для смеси третьего компонента со смесью первых двух по той же формуле
    return get_mixing_ponder_vansandern(epsi_12, epsi_3, _f1n + _f2n, _f3n)
