import numpy as np
from .ampld_par import *
from copy import deepcopy


def gauss(n: int, interval_type: int, print_info: bool = False) -> (np.array, np.array):
    """
    Расчет точек и весов для квадрурной формулы Гаусса. Формула и значения следуют и сверялись с данными из
    https://www.sciencedirect.com/topics/engineering/gaussian-quadrature-rule.

    :param n: Размер массива выходных данных, соответствует порядоку квардартуроной формулы
    :param interval_type: Тип интерваля для расчета,для 0 - интервал (-1, 1), для 1 - интервал от (0,1)
    :param print_info: Флаг вывода отладочной инорфмации
    :return: Два действительных массива размера N, содержащих точки и веса квадратурной формулы Гаусса
    """
    z_out = np.zeros(n)
    w_out = np.zeros(n)
    n_mod_2 = np.mod(n, 2)
    k = int(n / 2 + n_mod_2)

    for step in range(k):
        m = n - 1 - step
        if step == 0:
            x = 1. - 2. / ((n + 1.) * n)
        elif step == 1:
            x = (z_out[n - 1] - 1.) * 4 + z_out[n - 1]
        elif step == 2:
            x = (z_out[n - 2] - z_out[n - 1]) * 1.6 + z_out[n - 2]
        else:
            x = (z_out[m + 1] - z_out[m + 2]) * 3 + z_out[m + 3]
        if step == k and n_mod_2 == 0:
            x = 0

        num_iteration = 0
        check_precision = 1e-16

        while True:
            pb = 1
            num_iteration = num_iteration + 1
            if num_iteration > 100:
                check_precision = check_precision * 10
            pc = x
            dj = 1.

            for J in range(1, n):
                dj = dj + 1
                pa = pb
                pb = pc
                pc = x * pb + (x * pb - pa) * (dj - 1.) / dj

            pa = 1. / ((pb - x * pc) * n)
            pb = pa * pc * (1. - x * x)
            x = x - pb
            if abs(pb) <= check_precision * abs(x):
                break
        z_out[m] = x
        w_out[m] = pa * pa * (1. - x * x)
        if interval_type == 0:
            w_out[m] = 2. * w_out[m]

        if not (step == k and n_mod_2 == 1):
            z_out[step] = - z_out[m]
            w_out[step] = w_out[m]
    if print_info:
        print(f'***  POINTS AND WEIGHTS OF GAUSSIAN QUADRATURE FORMULA OF ', n, '-TH ORDER')

        for step in range(k):
            print(f'  X[{step}] = {-z_out[step]}, W[{step}] = {w_out[step]}')
        print(f' GAUSSIAN QUADRATURE FORMULA OF {n}-TH ORDER IS USED')

    if interval_type != 0:
        for step in range(n):
            z_out[step] = (1. + z_out[step]) / 2

    return z_out, w_out
