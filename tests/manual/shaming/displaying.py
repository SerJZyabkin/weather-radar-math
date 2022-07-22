import matplotlib.pyplot as plt
from membership import *
import numpy as np
from random import random, gauss

def plot_mf(ax, mf_to_plot, key):
    ax.plot(mf_to_plot, [0, 1, 1, 0], 'k-', linewidth=2)
    ax.legend([LABEL_NAMES[key]], loc='upper right')


def execute(min_class, max_class, product_type):
    fig, axes = plt.subplots(max_class - min_class + 1, 1, figsize=(8, 1.6 * (max_class - min_class + 1)))

    if product_type == 'Zh':
        mf_to_plot = MF_Zh
    elif product_type == 'Zdr':
        mf_to_plot = MF_Zdr
    else:
        print(f'Bad input key {product_type}.')
        quit()

    for ax, key in zip(axes, mf_to_plot.keys()):
        ax.set_xlim(XLIMS[product_type])
        ax.set_ylim([0, 1.2])
        ax.set_yticks([0, 0.5, 1])
        plot_mf(ax, mf_to_plot[key], key)
        ax.grid(True)

    axes[int(round((max_class - min_class) / 2))].set_ylabel('P(Zdr)', fontsize=14)
    axes[max_class].set_xlabel('Дифференциальная отражаемость, dB')

    plt.tight_layout()
    plt.show()

def get_rand_value():
    val = gauss(0, 0.3) + random() * 0.2 + 1.4

    return val

def exectute_learing():
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))

    ax.set_xlim(XLIMS['Zdr'])
    ax.set_ylim([0, 3])
    ax.set_yticks([0, 0.5, 1])
    plot_mf(ax, MF_Zdr['MR'], 'MR')
    ax.grid(True)

    ax.set_ylabel('P(Zdr)', fontsize=14)
    ax.set_xlabel('Дифференциальная отражаемость, dB')

    data = [get_rand_value() for _ in range(400)]

    counts, bins = np.histogram(data, 30)
    plt.hist(bins[:-1], bins, weights=counts/ 16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    execute(0, 7, 'Zdr')
    # exectute_learing()