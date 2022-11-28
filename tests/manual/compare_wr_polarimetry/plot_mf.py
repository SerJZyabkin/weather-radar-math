import matplotlib.pyplot as plt
import numpy as np
import os
from random import random
from scipy.stats import norm

from src.scattering_simulation.scattering_simulation.classification.membership_functions import \
    get_membership_function_info, get_products_list, get_classes_list, get_possible_data_range, get_class_label, \
    get_product_name_label

def extract_single_product_array(path_to_data: str, product_name: str):
    if product_name not in ['Zh', 'Zdr', 'Kdp', 'LDR', 'rohv']:
        print(f'Error extracting single array for {product_name}. Bad product name.')
        quit()

    with open(path_to_data, 'rb') as fid:
        raw_bytes = fid.read()
        data_array = np.frombuffer(raw_bytes, dtype=float)
        data_array = np.array(data_array.reshape([int(len(data_array) / 5), 5]))
        fid.close()

    if product_name == 'Zh':
        return data_array[:, 0]
    if product_name == 'Zdr':
        return data_array[:, 1]
    if product_name == 'LDR':
        return data_array[:, 2]
    if product_name == 'rohv':
        return data_array[:, 3]
    if product_name == 'Kdp':
        return data_array[:, 4]


def add_single_membership_function(axis, class_name: str, product_name: str):
    if class_name not in get_classes_list():
        print(f'Error plotting single membership function for {product_name} / {class_name}. Bad class name.')
        quit()
    if product_name not in get_products_list():
        print(f'Error plotting single membership function for {product_name} / {class_name}. Bad product name.')
        quit()

    mf_data = get_membership_function_info(class_name, product_name)
    axis.plot(mf_data, np.array([0, 1, 1, 0]), 'k', linewidth=2)


def add_single_histagramm(axis, file_name: str, product_name: str, num_bins: int):
    if product_name not in get_products_list():
        print(f'Error plotting single membership function for {product_name}. Bad product name.')
        quit()

    raw_data = extract_single_product_array(file_name, product_name)
    counts, bins = np.histogram(raw_data, bins=num_bins, range=get_possible_data_range(product_name))
    axis.hist(bins[:-1], bins, weights=counts / max(counts) * 0.9)

# path_to_data = 'E:/Git/weather-radar/weather-radar-math/bin1'
path_to_data = os.getcwd() + '/../../../bin_final'
path_to_data = os.getcwd() + '/../../../bin_final_1000'
path_to_data = os.getcwd() + '/../../../bin_final_2000'
os.chdir(path_to_data)


def plot_all_membershop_data(files_info: dict, product_name: str, num_bins: int = 200):
    fig, axis_list = plt.subplots(len(files_info), 1, figsize=(7 * 1.15, 9.3 * 1.15))

    x_data_range = get_possible_data_range(product_name)

    for cur_file_info, cur_axis in zip(files_info.items(), axis_list):
        class_name, file_name = cur_file_info
        add_single_histagramm(cur_axis, file_name, product_name, num_bins)
        add_single_membership_function(cur_axis, class_name, product_name)
        cur_axis.set_yticks([])
        cur_axis.set_ylabel(get_class_label(class_name))
        cur_axis.set_ylim([0, 1.3])
        cur_axis.set_xlim(x_data_range)

        cur_axis.set_xticks(np.linspace(x_data_range[0], x_data_range[1], 11))
        if cur_axis is not axis_list[-1]:
            cur_axis.set_xticklabels([])
        cur_axis.grid(True)

    axis_list[-1].set_xlabel(get_product_name_label(product_name))
    plt.show()


plot_all_membershop_data({'M': 'drizzle_mf.bin', 'LR': 'light_rain.bin', 'MR': 'medium_rain.bin',
                          'HR': 'heavy_rain.bin', 'LD': 'large_drops.bin', 'DS': 'dry_snow.bin',
                          'WS': 'wet_snow.bin', 'IC': 'ice_crystals.bin'}, 'Zh')