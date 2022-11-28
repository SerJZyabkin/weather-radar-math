import matplotlib.pyplot as plt
import numpy as np
import os
from random import random
from scipy.stats import norm

from src.scattering_simulation.scattering_simulation.classification.fuzzy_logic import FuzzyClassifier

NUM_POINTS_TO_PLOT = 10000


path_to_data = os.getcwd() + '/../../../bin_final_2000'
os.chdir(path_to_data)
classifier = FuzzyClassifier()


def get_detect_supercooled(IS_FUZZY):
    with open('cooled_water.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_array = np.frombuffer(raw_bytes, dtype=float)
        data_array = np.array(data_array.reshape([int(len(data_array) / 4), 4]))
        fid.close()

    correct_out = []
    wrong_out = []

    if IS_FUZZY:
        for step in range(len(data_array)):
            single_point = {'Zh': data_array[step, 0], 'Zdr': data_array[step, 1],
                            'LDR': data_array[step, 2], 'Kdp': data_array[step, 3]}

            class_out = classifier.classify(single_point)
            if class_out in ['IC', 'DS']:
                wrong_out.append([data_array[step, 1], data_array[step, 2]])
            else:
                correct_out.append([data_array[step, 1], data_array[step, 2]])


    else:
        for step in range(len(data_array)):
            if data_array[step, 2] > -27 and data_array[step, 1] > -0.4:
                correct_out.append([data_array[step, 1], data_array[step, 2]])
            else:
                wrong_out.append([data_array[step, 1], data_array[step, 2]])

    return np.array(correct_out), np.array(wrong_out)


def get_detect_crystalized(IS_FUZZY):
    with open('crystals.bin', 'rb') as fid:
        raw_bytes = fid.read()
        data_array = np.frombuffer(raw_bytes, dtype=float)
        data_array = np.array(data_array.reshape([int(len(data_array) / 4), 4]))
        fid.close()

    correct_out = []
    wrong_out = []

    if IS_FUZZY:
        for step in range(len(data_array)):
            single_point = {'Zh': data_array[step, 0], 'Zdr': data_array[step, 1],
                            'LDR': data_array[step, 2], 'Kdp': data_array[step, 3]}
            class_out = classifier.classify(single_point)
            if class_out in ['IC', 'DS']:
                correct_out.append([data_array[step, 1], data_array[step, 2]])
            else:
                wrong_out.append([data_array[step, 1], data_array[step, 2]])
    else:
        for step in range(len(data_array)):
            if data_array[step, 2] > -27 and data_array[step, 1] > -0.4:
                wrong_out.append([data_array[step, 1], data_array[step, 2]])
            else:
                correct_out.append([data_array[step, 1], data_array[step, 2]])

    return np.array(correct_out), np.array(wrong_out)


def plot_detect():
    fig, (axis, axis_2) = plt.subplots(2, 1, figsize=(8, 9))

    data_correct, data_wrong = get_detect_supercooled(False)
    axis.plot(data_correct[:NUM_POINTS_TO_PLOT, 0], data_correct[:NUM_POINTS_TO_PLOT, 1], '.', color='green')
    data_correct_2, data_wrong_2 = get_detect_crystalized(False)
    axis.plot(data_correct_2[:NUM_POINTS_TO_PLOT, 0], data_correct_2[:NUM_POINTS_TO_PLOT, 1], '+', color='green')
    axis.plot(data_wrong[:NUM_POINTS_TO_PLOT, 0], data_wrong[:NUM_POINTS_TO_PLOT, 1], '+', color='red')
    axis.plot(data_wrong_2[:NUM_POINTS_TO_PLOT, 0], data_wrong_2[:NUM_POINTS_TO_PLOT, 1], '.', color='red')


    print('Parametric method')
    print('Correct classify:', 100 * len(data_correct) / (len(data_correct) + len(data_wrong)))
    print('Correct nondetect:', 100 * len(data_correct_2) / (len(data_wrong_2) + len(data_correct_2)))

    data_correct, data_wrong = get_detect_supercooled(True)
    axis_2.plot(data_correct[:NUM_POINTS_TO_PLOT, 0], data_correct[:NUM_POINTS_TO_PLOT, 1], '.', color='green')
    data_correct_2, data_wrong_2 = get_detect_crystalized(True)
    axis_2.plot(data_correct_2[:NUM_POINTS_TO_PLOT, 0], data_correct_2[:NUM_POINTS_TO_PLOT, 1], '+', color='green')
    axis_2.plot(data_wrong[:NUM_POINTS_TO_PLOT, 0], data_wrong[:NUM_POINTS_TO_PLOT, 1], '+', color='red')
    axis_2.plot(data_wrong_2[:NUM_POINTS_TO_PLOT, 0], data_wrong_2[:NUM_POINTS_TO_PLOT, 1], '.', color='red')

    print('Fuzzy method')
    print('Correct classify:', 100 * len(data_correct) / (len(data_correct) + len(data_wrong)))
    print('Correct nondetect:', 100 * len(data_correct_2) / (len(data_wrong_2) + len(data_correct_2)))

    legends_list = ['Корректная классификация', 'Корректное необнаружение', 'Ошибочное необнаружение',
                    'Ошибочная классификация']
    axis_2.legend(legends_list, loc='lower right', ncol=1)
    #
    axis.set_xlim([-2.5, 3.5])
    axis.set_ylim([-70, -10])
    axis.grid(True)
    axis.set_xlabel('Zdr, dB')
    axis.set_ylabel('LDR, dB')

    axis_2.set_xlim([-2.5, 3.5])
    axis_2.set_ylim([-70, -10])
    axis_2.grid(True)
    axis_2.set_xlabel('Zdr, dB')
    axis_2.set_ylabel('LDR, dB')
    plt.show()


plot_detect()

