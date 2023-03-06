from ..calculation.tmatrix.ensemble import get_ensemble_generator
from json import load, dump
from os import path, remove, mkdir
import glob
import numpy as np
from math import fabs

_dbz_step = 2.
_min_dbz = -10
_max_dbz = 60.
_dbz_values = np.arange(_min_dbz, _max_dbz + _dbz_step, _dbz_step)
_dbz_bounds = [(_dbz_values[s], _dbz_values[s + 1]) for s in range(len(_dbz_values) - 1)]


def find_nearest(value):
    idx = np.searchsorted(_dbz_values, value, side="left")
    return idx
    # if idx > 0 and (idx == len(_dbz_values) or fabs(value - _dbz_values[idx-1]) < fabs(value - _dbz_values[idx])):
    #     return array[idx-1]
    # else:
    #     return array[idx]


class HistogramDataAccumulator:
    def __init__(self, path_to_data: str, json_info_file_name: str):
        self.config_file_name = json_info_file_name
        self.config = {}
        if path.isdir(path_to_data):
            if not path.isfile(json_info_file_name):
                #
                files_to_remove = glob.glob(path_to_data + '/*')

                if len(files_to_remove) > 0:
                    print(f'Папка {path_to_data} не пуста.\nФайл конфигурации {json_info_file_name} отсутствует.\n'
                          f'Для продолжения необходимо очистить папку.')
                    input_val = input('Введите "y" для удаления содержимого папки. Иначе операция будет прервана.\n')
                    if input_val == 'y' or input_val == 'Y':
                        for f in files_to_remove:
                            remove(f)
                        print(f'Папка {path_to_data} очищена от содержимого.')

                    else:
                        print('HistogramDataAccumulator.__init__: Операция прервана пользователем.')
                        quit()

                self.config['num_bounds'] = len(_dbz_bounds)
                bounds_info = {}
                for step in range(len(_dbz_bounds)):
                    bounds_info[step] = {'min_dbz': _dbz_bounds[step][0], 'max_dbz': _dbz_bounds[step][1]}
                self.config['num_bounds'] = len(_dbz_bounds)
                self.config['bounds_info'] = bounds_info

            else:
                with open(json_info_file_name, 'r') as outfile:
                    self.config = load(outfile)
                    outfile.close()

        else:
            mkdir(path_to_data)

    def execute(self, ensemble_name: str, wavelength: float):
        record_name = f'{ensemble_name}_{wavelength * 100}'

        current_config = {}
        if record_name in self.config.keys():
            current_config = self.config[record_name]
        else:
            current_config = {}
            for step in range(len(_dbz_bounds)):
                current_config[step] = 0

        product_generator = get_ensemble_generator(wavelength, ensemble_name)

        while True:
            out_values = next(product_generator)

            dbz = 1111 #out_values[0]

            indx = find_nearest(dbz)
            print(dbz, _dbz_bounds[indx])



            self.config[record_name] = current_config
            with open(self.config_file_name, 'w') as outfile:
                dump(self.config, outfile, indent=4)
                outfile.close()
            break
