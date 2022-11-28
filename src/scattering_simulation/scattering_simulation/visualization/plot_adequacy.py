import matplotlib.pyplot as plt
import numpy as np
import os
from random import random
from scipy.stats import norm

# path_to_data = 'E:/Git/weather-radar/weather-radar-math/bin1'
path_to_data = '/media/serj/Data/Git/repos/weather-radar-math/bin75'
os.chdir(path_to_data)

with open('ice_crystals.bin', 'rb') as fid:
    raw_bytes = fid.read()
    data_ic = np.frombuffer(raw_bytes, dtype=float)
    data_ic = np.array(data_ic.reshape([int(len(data_ic) / 5), 5]))
    for step in range(data_ic.shape[0]):
        data_ic[step, 0] = data_ic[step, 0] + random() * 6 - 10
    fid.close()

with open('dry_snow.bin', 'rb') as fid:
    raw_bytes = fid.read()
    data_ds = np.frombuffer(raw_bytes, dtype=float)
    data_ds = np.array(data_ds.reshape([int(len(data_ds) / 5), 5]))
    fid.close()

with open('wet_snow.bin', 'rb') as fid:
    raw_bytes = fid.read()
    data_ws = np.frombuffer(raw_bytes, dtype=float)
    data_ws = np.array(data_ws.reshape([int(len(data_ws) / 5), 5]))
    for step in range(data_ws.shape[0]):
        data_ws[step, 0] = data_ws[step, 0]  # + random() * 6 - 10
    fid.close()

with open('rain.bin', 'rb') as fid:
    raw_bytes = fid.read()
    data_r = np.frombuffer(raw_bytes, dtype=float)
    data_r = np.array(data_r.reshape([int(len(data_r) / 5), 5]))
    for step in range(data_r.shape[0]):
        data_r[step, 0] = data_r[step, 0]  # + random() * 6 - 10
    fid.close()

with open('drizzle.bin', 'rb') as fid:
    raw_bytes = fid.read()
    data_drizzle = np.frombuffer(raw_bytes, dtype=float)
    data_drizzle = data_drizzle.reshape([int(len(data_drizzle) / 5), 5])
    fid.close()


def get_beta_membership(mean_left, mean_right, dispersion, max_val_y = 1.):
    multi = max_val_y / norm.pdf(mean_right, mean_right, dispersion)
    def beta_func(val):
        if val < mean_left:
            return norm.pdf(val, mean_left, dispersion) * multi
        elif val > mean_right:
            return norm.pdf(val, mean_right, dispersion) * multi
        else:
            return max_val_y
    return beta_func

# sigma = np.var(data_drizzle[:, 0])
# mu = np.mean(data_drizzle[:, 0])


def plot_reflectivity():
    fig, ax = plt.subplots(5, 1, figsize=(7, 16), dpi=100)

    ax[0].hist(data_drizzle[:, 0], bins=100)
    x_arr = np.linspace(0, 70, 1000)
    beta_func = get_beta_membership(-10, 21, 2, 30)
    ax[0].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[0].set_xlim([0, 70])
    ax[0].set_ylim([0, 42])
    ax[0].set_yticks([])
    ax[0].set_ylabel('Морось')

    ax[1].hist(data_r[:, 0], bins=100)
    beta_func = get_beta_membership(28, 57, 2, 20)
    ax[1].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[1].set_xlim([0, 70])
    ax[1].set_yticks([])
    ax[1].set_ylabel('Дождь')

    ax[2].hist(data_ds[:, 0], bins = 100)
    beta_func = get_beta_membership(-10, 35, 2, 30)
    ax[2].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[2].set_xlim([0, 70])
    ax[2].set_ylim([0, 42])
    ax[2].set_yticks([])
    ax[2].set_ylabel('Сухой снег')

    ax[3].hist(data_ic[:, 0], bins = 100)
    beta_func = get_beta_membership(-10, 24, 2, 25)
    ax[3].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[3].set_xlim([0, 70])
    ax[3].set_yticks([])
    ax[3].set_ylabel('Кристаллы льда')

    ax[4].hist(data_ws[:, 0], bins = 100)
    beta_func = get_beta_membership(-10, 35, 2, 25)
    ax[4].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[4].set_xlim([0, 70])
    ax[4].set_ylim([0, 35])
    ax[4].set_yticks([])
    ax[4].set_ylabel('Мокрый снег')
    ax[4].set_xlabel('Радиолокационная отражаемость Zh, dBZ')
    plt.show()


def plot_zdr():
    fig, ax = plt.subplots(5, 1, figsize=(7, 16), dpi=100)

    ax[0].hist(data_drizzle[:, 1], bins=200, range=(-1, 6))
    x_arr = np.linspace(-1, 6, 1000)
    beta_func = get_beta_membership(0.2, 0.6, 0.2, 200)
    ax[0].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[0].set_xlim([-1, 6])
    ax[0].set_ylim([0, 250])
    ax[0].set_yticks([])
    ax[0].set_ylabel('Морось')

    ax[1].hist(data_r[:, 1], bins=200, range=(-1, 6))
    beta_func = get_beta_membership(0.5, 3.0, 0.2, 20)
    ax[1].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[1].set_xlim([-1, 6])
    ax[1].set_ylim([0, 28])
    ax[1].set_yticks([])
    ax[1].set_ylabel('Дождь')

    ax[2].hist(data_ds[:, 1], bins=200, range=(-1, 6))
    beta_func = get_beta_membership(0.2, 0.6, 0.3, 40)
    ax[2].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[2].set_xlim([-1, 6])
    ax[2].set_ylim([0, 52])
    ax[2].set_yticks([])
    ax[2].set_ylabel('Сухой снег')

    ax[3].hist(data_ic[:, 1], bins=200, range=(-1, 6))
    beta_func = get_beta_membership(0.2, 4.7, 0.1, 15)
    ax[3].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[3].set_xlim([-1, 6])
    ax[3].set_ylim([0, 23])
    ax[3].set_yticks([])
    ax[3].set_ylabel('Кристаллы льда')

    ax[4].hist(data_ws[:, 1], bins=200, range=(-1, 6))
    beta_func = get_beta_membership(0.2, 3.2, 0.1, 25)
    ax[4].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[4].set_xlim([-1, 6])
    ax[4].set_ylim([0, 36])
    ax[4].set_yticks([])
    ax[4].set_ylabel('Мокрый снег')
    ax[4].set_xlabel('Дифференциальная отражаемость Zdr, dB')

    plt.show()


def plot_ldr():
    fig, ax = plt.subplots(5, 1, figsize=(7, 16), dpi=100)

    ax[0].hist(data_drizzle[:, 2], bins=200, range=(-50, 0))
    x_arr = np.linspace(-50, 0, 1000)
    beta_func = get_beta_membership(-60., -37., 1.4, 15)
    ax[0].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[0].set_xlim([-50, -10])
    ax[0].set_ylim([0, 21])
    ax[0].set_yticks([])
    ax[0].set_ylabel('Морось')

    ax[1].hist(data_r[:, 2], bins=200, range=(-50, 0))
    beta_func = get_beta_membership(-40., -27.0, 2, 20)
    ax[1].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[1].set_xlim([-50, -10])
    ax[1].set_ylim([0, 25])
    ax[1].set_yticks([])
    ax[1].set_ylabel('Дождь')

    ax[2].hist(data_ds[:, 2], bins=200, range=(-50, 0))
    beta_func = get_beta_membership(-50, -36, 1.8, 14)
    ax[2].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[2].set_xlim([-50, -10])
    ax[2].set_ylim([0, 18])
    ax[2].set_yticks([])
    ax[2].set_ylabel('Сухой снег')

    ax[3].hist(data_ic[:, 2], bins=200, range=(-50, 0))
    beta_func = get_beta_membership(-37, -25, 2, 21)
    ax[3].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[3].set_xlim([-50, -10])
    ax[3].set_ylim([0, 27])
    ax[3].set_yticks([])
    ax[3].set_ylabel('Кристаллы льда')

    ax[4].hist(data_ws[:, 2], bins=200, range=(-50, 0))
    beta_func = get_beta_membership(-26, -17, 2, 35)
    ax[4].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[4].set_xlim([-50, -10])
    ax[4].set_ylim([0, 47])
    ax[4].set_yticks([])
    ax[4].set_ylabel('Мокрый снег')
    ax[4].set_xlabel('Линейное деполяризационное отношение Ldr, dB')

    plt.show()


def plot_rohv():
    fig = plt.figure(figsize=(9, 6), dpi=100)
    ax00 = fig.add_subplot(321)
    ax01 = fig.add_subplot(322)
    ax10 = fig.add_subplot(323)
    ax11 = fig.add_subplot(324)
    ax21 = fig.add_subplot(313)

    ax00.hist(data_drizzle[:, 3], bins=90, range=(0.91, 1.0))
    x_arr = np.linspace(0.91, 1.01, 1000)
    beta_func = get_beta_membership(1, 1.1, 0.002, 1000)
    ax00.plot(x_arr, [beta_func(x) for x in x_arr])
    ax00.set_xlim([0.91, 1.0])
    ax00.set_yticks([])
    ax00.set_ylabel('Морось')

    ax10.hist(data_r[:, 3], bins=90, range=(0.91, 1.0))
    beta_func = get_beta_membership(0.99, 1.1, 0.004, 400)
    ax10.plot(x_arr, [beta_func(x) for x in x_arr])
    ax10.set_xlim([0.91, 1.0])
    ax10.set_ylim([0, 480])
    ax10.set_yticks([])
    ax10.set_ylabel('Дождь')

    ax01.hist(data_ds[:, 3], bins=90, range=(0.91, 1.0))
    beta_func = get_beta_membership(0.99, 1.1, 0.004, 800)
    ax01.plot(x_arr, [beta_func(x) for x in x_arr])
    ax01.set_xlim([0.91, 1.0])
    ax01.set_ylim([0, 980])
    ax01.set_yticks([])
    ax01.set_ylabel('Сухой снег')

    ax11.hist(data_ic[:, 3], bins=90, range=(0.91, 1.0))
    beta_func = get_beta_membership(0.96, 1.1, 0.01, 220)
    ax11.plot(x_arr, [beta_func(x) for x in x_arr])
    ax11.set_xlim([0.91, 1.0])
    ax11.set_ylim([0, 295])
    ax11.set_yticks([])
    ax11.set_ylabel('Кристаллы льда')

    ax21.hist(data_ws[:, 3], bins=90, range=(0.91, 1.0))
    beta_func = get_beta_membership(0.82, 0.93, 0.01, 35)
    ax21.plot(x_arr, [beta_func(x) for x in x_arr])
    ax21.set_xlim([0.91, 1.0])
    ax21.set_ylim([0, 45])
    ax21.set_yticks([])
    ax21.set_ylabel('Мокрый снег')
    ax21.set_xlabel('Коэффициент корреляции ρhv')

    plt.show()


def plot_kdp():
    fig, ax = plt.subplots(5, 1, figsize=(7, 16), dpi=100)

    ax[0].hist(data_drizzle[:, 4], bins=200, range=(-1, 4))
    x_arr = np.linspace(-1, 4, 1000)
    beta_func = get_beta_membership(0.02, 0.02, 0.02, 1000)
    ax[0].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[0].set_xlim([-1, 4])
    ax[0].set_yticks([])
    ax[0].set_ylabel('Морось')


    ax[1].hist(data_r[:, 4], bins=200, range=(-1, 4))
    beta_func = get_beta_membership(0.1, 27.0, 0.1, 18)
    ax[1].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[1].set_xlim([-1, 4])
    ax[1].set_ylim([0, 20])
    ax[1].set_yticks([])
    ax[1].set_ylabel('Дождь')


    ax[2].hist(data_ds[:, 4], bins=200, range=(-1, 4))
    beta_func = get_beta_membership(-0.1, 0.2, 0.2, 17)
    ax[2].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[2].set_xlim([-1, 4])
    ax[2].set_ylim([0, 20])
    ax[2].set_yticks([])
    ax[2].set_ylabel('Сухой снег')


    ax[3].hist(data_ic[:, 4], bins=200, range=(-1, 4))
    beta_func = get_beta_membership(0.1, 0.3, 0.2, 17)
    ax[3].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[3].set_xlim([-1, 4])
    ax[3].set_ylim([0, 20])
    ax[3].set_yticks([])
    ax[3].set_ylabel('Кристаллы льда')

    ax[4].hist(data_ws[:, 4], bins=200, range=(-1, 4))
    beta_func = get_beta_membership(0.1, 2.4, 0.1, 17)
    ax[4].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[4].set_xlim([-1, 4])
    ax[4].set_ylim([0, 20])
    ax[4].set_yticks([])
    ax[4].set_ylabel('Мокрый снег')
    ax[4].set_xlabel('Коэффициент корреляции Kdp, °')

    plt.show()

# plot_reflectivity()
# plot_zdr()
# plot_ldr()
plot_rohv()
# plot_kdp()

