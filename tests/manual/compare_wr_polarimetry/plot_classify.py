import matplotlib.pyplot as plt
import numpy as np
import os
from random import random
from scipy.stats import norm

from src.scattering_simulation.scattering_simulation.classification.fuzzy_logic import FuzzyClassifier

NUM_POINTS_TO_PLOT = 10000

path_to_data = os.getcwd() + '/../../../bin_classify'
os.chdir(path_to_data)
classifier = FuzzyClassifier()


def get_classify_supercooled(fname, IS_FUZZY):
    with open(fname, 'rb') as fid:
        raw_bytes = fid.read()
        data_array = np.frombuffer(raw_bytes, dtype=float)
        data_array = np.array(data_array.reshape([int(len(data_array) / 4), 4]))
        fid.close()

    correct_out = []
    wrong_out = []

    if IS_FUZZY:
        for step in range(len(data_array) - 1):
            single_point = {'Zh': data_array[step, 0], 'Zdr': data_array[step, 1],
                            'LDR': data_array[step, 2], 'Kdp': data_array[step, 3]}

            class_out = classifier.classify(single_point)
            if class_out in ['IC', 'DS']:
                wrong_out.append([data_array[step, 0], data_array[step, 1]])
            else:
                correct_out.append([data_array[step, 0], data_array[step, 1]])

    else:
        for step in range(len(data_array) - 1):
            if data_array[step, 2] > -27 and data_array[step, 1] > -0.4:
                correct_out.append([data_array[step, 0], data_array[step, 1]])
            else:
                wrong_out.append([data_array[step, 0], data_array[step, 1]])

    return np.array(correct_out), np.array(wrong_out)


def get_classify_crystalized(fname, IS_FUZZY):
    with open(fname, 'rb') as fid:
        raw_bytes = fid.read()
        data_array = np.frombuffer(raw_bytes, dtype=float)
        data_array = np.array(data_array.reshape([int(len(data_array) / 4), 4]))
        fid.close()

    correct_out = []
    wrong_out = []

    if IS_FUZZY:
        for step in range(len(data_array) - 1):
            single_point = {'Zh': data_array[step, 0], 'Zdr': data_array[step, 1],
                            'LDR': data_array[step, 2], 'Kdp': data_array[step, 3]}
            class_out = classifier.classify(single_point)
            if class_out in ['IC', 'DS']:
                correct_out.append([data_array[step, 0], data_array[step, 1]])
            else:
                wrong_out.append([data_array[step, 0], data_array[step, 1]])
    else:
        for step in range(len(data_array) - 1):
            if data_array[step, 2] > -27 and data_array[step, 1] > -0.4:
                wrong_out.append([data_array[step, 0], data_array[step, 1]])
            else:
                correct_out.append([data_array[step, 0], data_array[step, 1]])

    return np.array(correct_out), np.array(wrong_out)


def plot_classify():
    fig, (axis, axis2) = plt.subplots(2, 1, figsize=(8, 8))

    data_correct_w5, data_wrong_w5 = get_classify_supercooled('cw_5.bin', True)
    data_correct_i5, data_wrong_i5 = get_classify_crystalized('ic_5.bin', True)

    data_correct_w15, data_wrong_w15 = get_classify_supercooled('cw_15.bin', True)
    data_correct_i15, data_wrong_i15 = get_classify_crystalized('ic_15.bin', True)

    data_correct_w25, data_wrong_w25 = get_classify_supercooled('cw_25.bin', True)
    data_correct_i25, data_wrong_i25 = get_classify_crystalized('ic_25.bin', True)
    #
    # data_correct_w = np.array(list(data_correct_w5) + list(data_correct_w15) + list(data_correct_w25))
    # data_wrong_w = np.array(list(data_wrong_w5) + list(data_wrong_w15) + list(data_wrong_i25))

    data_correct_w = np.array(list(data_correct_w5) + list(data_correct_w15) + list(data_correct_w25))
    data_wrong_w = np.array(list(data_wrong_w5) + list(data_wrong_w15) + list(data_wrong_w25))

    data_correct_i = np.array(list(data_correct_i5) + list(data_correct_i15) + list(data_correct_i25))
    data_wrong_i = np.array(list(data_wrong_i5) + list(data_wrong_i15) + list(data_wrong_i25))

    axis.plot(data_correct_w[:NUM_POINTS_TO_PLOT, 0], data_correct_w[:NUM_POINTS_TO_PLOT, 1], '.', color='green')
    axis.plot(data_wrong_w[:NUM_POINTS_TO_PLOT, 0], data_wrong_w[:NUM_POINTS_TO_PLOT, 1], '+', color='red')

    axis2.plot(data_correct_i[:NUM_POINTS_TO_PLOT, 0], data_correct_i[:NUM_POINTS_TO_PLOT, 1], '.', color='green')
    axis2.plot(data_wrong_i[:NUM_POINTS_TO_PLOT, 0], data_wrong_i[:NUM_POINTS_TO_PLOT, 1], '+', color='red')

    axis2.plot([15, 15], [-10, 10], '--k', linewidth=3)
    axis2.plot([5, 5], [-10, 10], '--k', linewidth=3)
    axis2.plot([25, 25], [-10, 10], '--k', linewidth=3)

    axis.plot([15, 15], [-10, 10], '--k', linewidth=3)
    axis.plot([5, 5], [-10, 10], '--k', linewidth=3)
    axis.plot([25, 25], [-10, 10], '--k', linewidth=3)

    print('Fuzzy method')
    print('\ndBZ 5 - 15:')
    print('Correct classify:', 100 * len(data_correct_w5) / (len(data_correct_w5) + len(data_wrong_w5)))
    print('Correct nondetect:',100 - 100 * len(data_correct_i5) / (len(data_wrong_i5) + len(data_correct_i5)))

    print('\ndBZ 15 - 25:')
    print('Correct classify:', 100 * len(data_correct_w15) / (len(data_correct_w15) + len(data_wrong_w15)))
    print('Correct nondetect:', 100 - 100 * len(data_correct_i15) / (len(data_wrong_i15) + len(data_correct_i15)))


    print('\ndBZ 25 +:')
    print('Correct classify:', 100 * len(data_correct_w25) / (len(data_correct_w25) + len(data_wrong_w25)))
    print('Correct nondetect:', 100 - 100 * len(data_correct_i25) / (len(data_wrong_i25) + len(data_correct_i25)))

    legends_list = ['Корректная классификация', 'Ошибочная классификация']
    axis2.legend(legends_list, loc='lower right', ncol=1)
    #
    axis.set_ylim([-2.5, 3.5])
    axis.set_xlim([0, 70])
    axis.grid(True)
    axis.set_ylabel('Zdr, dB')
    axis.set_xlabel('Zh, dBZ')

    axis2.set_ylim([-2.5, 3.5])
    axis2.set_xlim([0, 70])
    axis2.grid(True)
    axis2.set_ylabel('Zdr, dB')
    axis2.set_xlabel('Zh, dBZ')
    plt.show()


plot_classify()

