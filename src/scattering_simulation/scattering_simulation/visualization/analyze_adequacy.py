import matplotlib.pyplot as plt
import numpy as np
import os
from random import random
from scipy.stats import norm

path_to_data = 'E:/Git/weather-radar/weather-radar-math/bin2'  # '/media/serj/Data/Git/repos/weather-radar-math/bin'
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
    fig, ax = plt.subplots(5, 1)

    ax[0].hist(data_drizzle[:, 0], bins = 100)
    x_arr = np.linspace(0, 70, 1000)
    beta_func = get_beta_membership(-10, 21, 2, 20)
    ax[0].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[0].set_xlim([0, 70])

    ax[1].hist(data_r[:, 0], bins = 100)
    beta_func = get_beta_membership(28, 57, 2, 20)
    ax[1].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[1].set_xlim([0, 70])

    ax[2].hist(data_ds[:, 0], bins = 100)
    beta_func = get_beta_membership(-10, 35, 2, 20)
    ax[2].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[2].set_xlim([0, 70])

    ax[3].hist(data_ic[:, 0], bins = 100)
    beta_func = get_beta_membership(-10, 24, 2, 4)
    ax[3].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[3].set_xlim([0, 70])

    ax[4].hist(data_ws[:, 0], bins = 100)
    beta_func = get_beta_membership(-10, 35, 2, 20)
    ax[4].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[4].set_xlim([0, 70])

    plt.show()

def plot_zdr():
    fig, ax = plt.subplots(5, 1)

    ax[0].hist(data_drizzle[:, 1], bins=100)
    x_arr = np.linspace(-1, 6, 1000)
    beta_func = get_beta_membership(0.2, 0.6, 0.2, 2)
    ax[0].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[0].set_xlim([-1, 6])

    ax[1].hist(data_r[:, 1], bins=100)
    beta_func = get_beta_membership(0.5, 3.3, 0.2, 20)
    ax[1].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[1].set_xlim([-1, 6])

    ax[2].hist(data_ds[:, 1], bins=100)
    beta_func = get_beta_membership(0.2, 0.6, 0.3, 20)
    ax[2].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[2].set_xlim([-1, 6])

    ax[3].hist(data_ic[:, 1], bins=100)
    beta_func = get_beta_membership(0.2, 4.7, 0.1, 4)
    ax[3].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[3].set_xlim([-1, 6])

    ax[4].hist(data_ws[:, 1], bins=100)
    beta_func = get_beta_membership(0.2, 3.0, 0.1, 32)
    ax[4].plot(x_arr, [beta_func(x) for x in x_arr])
    ax[4].set_xlim([-1, 6])

    plt.show()


# plot_reflectivity()
plot_zdr()