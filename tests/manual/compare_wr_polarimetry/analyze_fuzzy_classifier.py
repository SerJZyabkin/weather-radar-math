import matplotlib.pyplot as plt
import numpy as np
import os
from random import random
from scipy.stats import norm
from random import uniform
from src.scattering_simulation.scattering_simulation.classification.fuzzy_logic import FuzzyClassifier
from src.scattering_simulation.scattering_simulation.calculation.tmatrix.spheroid import SpheroidModel
from src.scattering_simulation.scattering_simulation.classification.membership_functions import *

# init_seed = 10
# wavelength = 0.15
#
# model_rain = SpheroidModel('ice_crystals', wavelength, 'general', init_seed)
# classifier = FuzzyClassifier()
# for _ in range(10):
#     Zh, Zdr, LDR, _, Kdp = model_rain.get_products(-5, 10)
#     class_out = classifier.classify({'Zh': Zh, 'Zdr': Zdr, 'LDR': LDR, 'Kdp': Kdp})
#     print(class_out)
path_to_data = os.getcwd() + '/../../../bin_final_2000'
os.chdir(path_to_data)
classifier = FuzzyClassifier()

paths_to_data = {'M': 'drizzle_mf.bin', 'LR': 'light_rain.bin', 'MR': 'medium_rain.bin',
                 'HR': 'heavy_rain.bin', 'LD': 'large_drops.bin', 'DS': 'dry_snow.bin',
                 'WS': 'wet_snow.bin', 'IC': 'ice_crystals.bin'}

INIT_SEED = 10
WAVELENGTH = 0.0325
possible_classes = list(get_classes_list())
result_values = np.zeros([len(possible_classes), len(possible_classes)])
result_values_weighted = np.zeros([len(possible_classes), len(possible_classes)])


def get_dict_generator_for_classifier(path_to_data: str):
    with open(path_to_data, 'rb') as fid:
        raw_bytes = fid.read()
        data_array = np.frombuffer(raw_bytes, dtype=float)
        data_array = np.array(data_array.reshape([int(len(data_array) / 5), 5]))
        fid.close()

    def generator_product():
        for single_data in data_array:
            yield {'Zh': single_data[0], 'Zdr': single_data[1], 'LDR': single_data[2], 'Kdp': single_data[4]}

        while True:
            yield None

    return generator_product()


NUM_ITERATIONS = [0] * len(possible_classes)

for step_class_name_in in range(len(possible_classes)):
    counter_iters = 0
    product_gen = get_dict_generator_for_classifier(paths_to_data[possible_classes[step_class_name_in]])
    next_product = next(product_gen)

    while next_product is not None:
        NUM_ITERATIONS[step_class_name_in] += 1
        current_class_name = classifier.classify(next_product)
        current_weighted_class_name = classifier.classify_weighted(next_product)
        for step_class_name_out in range(len(possible_classes)):
            if possible_classes[step_class_name_out] == current_class_name:
                result_values[step_class_name_in, step_class_name_out] += 1
            if possible_classes[step_class_name_out] == current_weighted_class_name:
                result_values_weighted[step_class_name_in, step_class_name_out] += 1

        next_product = next(product_gen)

def prepare(result_values, classes_list, current_class):

    results = list(result_values)
    classes = list(classes_list)
    cur_index = classes.index(current_class)
    classes.pop(cur_index)
    results.pop(cur_index)

    out_dict = {}
    while len(classes) > 0:
        max_value = max(results)
        max_index = list(results).index(max_value)
        out_dict[classes[max_index]] = max_value
        classes.pop(max_index)
        results.pop(max_index)
    return out_dict

print('Results:')
for step in range(len(possible_classes)):
    print(f'Classifying {possible_classes[step]}')
    water_accum = 0
    ice_accum = 0
    for step_accum in range(len(possible_classes)):
        if is_class_supercooled(possible_classes[step_accum]):
            water_accum += result_values[step, step_accum]
        else:
            ice_accum += result_values[step, step_accum]

    if is_class_supercooled(possible_classes[step]):
        print(f'    {possible_classes[step]}: {100 * result_values[step,step] / NUM_ITERATIONS[step]} %  '
              f'/ {100 * water_accum / NUM_ITERATIONS[step]} %')
    else:
        print(f'    {possible_classes[step]}: {100 * result_values[step,step] / NUM_ITERATIONS[step]} %  '
              f'/ {100 * ice_accum / NUM_ITERATIONS[step]} %')

    water_accum_weighted = 0
    ice_accum_weighted = 0
    for step_accum in range(len(possible_classes)):
        if is_class_supercooled(possible_classes[step_accum]):
            water_accum_weighted += result_values_weighted[step, step_accum]
        else:
            ice_accum_weighted += result_values_weighted[step, step_accum]
    if is_class_supercooled(possible_classes[step]):
        print(f'    {possible_classes[step]}: {100 * result_values_weighted[step,step] / NUM_ITERATIONS[step]} %  '
              f'/ {100 * water_accum_weighted / NUM_ITERATIONS[step]} %')
    else:
        print(f'    {possible_classes[step]}: {100 * result_values_weighted[step,step] / NUM_ITERATIONS[step]} %  '
              f'/ {100 * ice_accum_weighted / NUM_ITERATIONS[step]} %')

    other_info = prepare(result_values[step, :], possible_classes, possible_classes[step])
    info_string = '    '
    for cur_class, cur_value in other_info.items():
        if cur_value > 0.:
            info_string += f'{cur_class}: {100 * cur_value / NUM_ITERATIONS[step]} %  |  '
    print('\n' + info_string)

    other_info = prepare(result_values_weighted[step, :], possible_classes, possible_classes[step])
    info_string = '    '
    for cur_class, cur_value in other_info.items():
        if cur_value > 0.:
            info_string += f'{cur_class}: {100 * cur_value /  NUM_ITERATIONS[step]} %  |  '
    print(info_string + '\n')

classes_labels = [get_class_label(class_name) for class_name in possible_classes]



results = {class_name: current_result for class_name, current_result in zip(classes_labels, result_values_weighted)}

labels = list(results.keys())
data = np.array(list(results.values()))
data_cum = data.cumsum(axis=1)
category_colors = plt.colormaps['RdYlGn'](
    np.linspace(0.15, 0.85, data.shape[1]))

fig, ax = plt.subplots(figsize=(9.2, 7))
ax.invert_yaxis()
ax.xaxis.set_visible(False)
ax.set_xlim(0, np.sum(data, axis=1).max())

for i, (colname, color) in enumerate(zip(classes_labels, category_colors)):
    widths = data[:, i]
    starts = data_cum[:, i] - widths
    rects = ax.barh(labels, widths, left=starts, height=0.5,
                    label=colname, color=color)

    r, g, b, _ = color
    text_color = 'white' if r * g * b < 0.5 else 'darkgrey'

ax.legend(ncol=len(classes_labels), bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')

plt.show()


