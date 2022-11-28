from typing import Callable

_CLASS_NAMES = {'M': 'Морось', 'LR': 'Слабый дождь', 'MR': 'Умеренный дождь', 'HR': 'Ливневый дождь', 'LD':
                '"Крупные капли"', 'DS': 'Сухой снег', 'WS': 'Мокрый снег', 'IC': 'Ориентированные кристаллы льда'}

_PRODUCT_NAMES = {'Zh': 'Радиолокационная отражаемость, дБZ', 'Zdr': 'Дифференциальная отражаемость, дБ',
                  'LDR': 'Линейное деполяризационное отношение, дБ', 'Kdp': 'Удельная дифференциальная фаза, °'}

_CLASS_LABELS = {'M': 'M', 'LR': 'СД', 'MR': 'УД', 'HR': 'ЛД', 'LD': 'КК', 'DS': 'СС', 'WS': 'МС', 'IC': 'ОКЛ'}

_MODEL_NAMES = {'M': ('drizzle', 'general'), 'LR': ('light_rain', 'general'), 'MR': ('medium_rain', 'general'),
                'HR': ('heavy_rain', 'general'), 'LD': ('large_drops', 'general'), 'DS': ('dry_snow', 'general'),
                'WS': ('wet_snow', 'general'), 'IC': ('ice_crystals', 'general')}

_PRODUCT_DATA_RANGES = {'Zh': (-10, 60), 'Zdr': (-2.5, 3.5), 'LDR': (-70, -10), 'Kdp': (-1., 9)}

_MF_Zh = {'M': [-15, -10, 20, 30], 'LR': [5, 22, 30, 40], 'MR': [34, 40, 46, 51],
          'HR': [36, 46, 70, 75], 'LD': [25, 38, 47, 52], 'DS': [-8, -1, 20, 42],
          'WS': [0, 12, 30, 48], 'IC': [-50, -40, 15, 32]}

_MF_Zdr = {'M': [-0.1, -0.05, 0.05, 0.1], 'LR': [0., 0.2, 0.75, 0.85], 'MR': [0.4, 0.8, 2., 2.3],
           'HR': [0.8, 2.0, 4, 4.5], 'LD': [0.4, 1.4, 4, 4.5], 'DS': [-0.2, 0, 0.6, 0.8],
           'WS': [1.5, 1.9, 2.8, 3.2], 'IC': [-3.5, -3.0, 0.8, 3]}

_MF_LDR = {'M': [-100, -90, -60, -51], 'LR': [-57, -45, -36, -28], 'MR': [-41, -36, -28, -24],
           'HR': [-36, -26, -24, -23], 'LD': [-38, -28, -25, -21], 'DS': [-75, -60, -38, -33],
           'WS': [-30, -23, -20, -17], 'IC': [-34, -18, -5, 0]}

_MF_Kdp = {'M': [-0.08, -0.04, 0.02, 0.05], 'LR': [-0.1, 0.1, 1.8, 3.1], 'MR': [1.4, 2.1, 25, 31],
           'HR': [1.4, 2.1, 44, 45.1], 'LD': [-0.2, 0.6, 4.8, 8.2], 'DS': [-0.2, 0, 0.2, 0.6],
           'WS': [-0.1, 0., 0.05, 3], 'IC': [-2, -1, 0.1, 0.3]}

_MEMBERSHIP_FUNCTIONS = {'Zh': _MF_Zh, 'Zdr': _MF_Zdr, 'LDR': _MF_LDR, 'Kdp': _MF_Kdp}


# _WEIGHTS_Zh = {'M': 1,     'LR': 1,   'MR': 1,   'HR': 1,   'LD': 1,   'DS': 1,   'WS': 1,  'IC': 1}
# _WEIGHTS_Zdr = {'M': 0.8,  'LR': 1,   'MR': 1,   'HR': 1,   'LD': 1,   'DS': 0.6, 'WS': 0.8,  'IC': 0.8}
# _WEIGHTS_LDR = {'M': 0.45, 'LR': 0.3, 'MR': 0.3, 'HR': 0.3, 'LD': 0.3, 'DS': 0.5, 'WS': 0.4, 'IC': 0.4}
# _WEIGHTS_Kdp = {'M': 0.45, 'LR': 0.5, 'MR': 0.5, 'HR': 0.5, 'LD': 0.5, 'DS': 0.6,   'WS': 0.5,  'IC': 0.6}


_WEIGHTS_Zh = {'M': 1,     'LR': 1,   'MR': 1,   'HR': 0.9,   'LD': 0.9,   'DS': 1,   'WS': 0.9,  'IC': 1}
_WEIGHTS_Zdr = {'M': 0.9,  'LR': 1,   'MR': 1,   'HR': 1,   'LD': 1,   'DS': 0.8, 'WS': 0.9,  'IC': 1.1}
_WEIGHTS_LDR = {'M': 0.95, 'LR': 0.8, 'MR': 0.8, 'HR': 0.8, 'LD': 0.8, 'DS': 1, 'WS': 0.9, 'IC': 0.9}
_WEIGHTS_Kdp = {'M': 0.85, 'LR': 0.9, 'MR': 0.9, 'HR': 0.9, 'LD': 0.9, 'DS': 0.7,   'WS': 0.6,  'IC': 0.9}
#
# _WEIGHTS_Zh = {'M': 1,     'LR': 1,   'MR': 1,   'HR': 1,   'LD': 1,   'DS': 1,   'WS': 1,  'IC': 1}
# _WEIGHTS_Zdr = {'M': 1,     'LR': 1,   'MR': 1,   'HR': 1,   'LD': 1,   'DS': 1,   'WS': 1,  'IC': 1}
# _WEIGHTS_LDR = {'M': 1,     'LR': 1,   'MR': 1,   'HR': 1,   'LD': 1,   'DS': 1,   'WS': 1,  'IC': 1}
# _WEIGHTS_Kdp = {'M': 1,     'LR': 1,   'MR': 1,   'HR': 1,   'LD': 1,   'DS': 1,   'WS': 1,  'IC': 1}

_SUPERCOOLED_WATER = ('M', 'LR', 'MR', 'HR', 'LD', 'WS')
_CRYSTALLIZED_WATER = list(_CLASS_NAMES.keys())
for val in _SUPERCOOLED_WATER:
    _CRYSTALLIZED_WATER.remove(val)
_CRYSTALLIZED_WATER = tuple(_CRYSTALLIZED_WATER)

_PRODUCT_WEIGHTS = {'Zh': _WEIGHTS_Zh, 'Zdr': _WEIGHTS_Zdr, 'LDR': _WEIGHTS_LDR, 'Kdp': _WEIGHTS_Kdp}

_CLASS_TOTAL_WEIGHTS = {}
for key_class in _CLASS_NAMES.keys():
    total_weight = 0
    for key_product in _PRODUCT_NAMES.keys():
        total_weight += _PRODUCT_WEIGHTS[key_product][key_class]
    _CLASS_TOTAL_WEIGHTS[key_class] = total_weight

def get_membership_function_info(class_name: str, product_name: str) -> list:
    if product_name in _MEMBERSHIP_FUNCTIONS.keys():
        if class_name in _MEMBERSHIP_FUNCTIONS[product_name].keys():
            return _MEMBERSHIP_FUNCTIONS[product_name][class_name]
        else:
            print(f'membership_functions.get_membership: received unknown class name \'{class_name}\'. Exiting!')
            quit()
    else:
        print(f'membership_functions.get_membership: received unknown product name \'{product_name}\'. Exiting!')
        quit()


def _get_membership_function(class_name: str, product_name: str) -> Callable[[float], complex]:
    mf_info = get_membership_function_info(class_name, product_name)

    def out_function(in_val: float) -> float:
        delta_left = 1 / (mf_info[1] - mf_info[0])
        delta_right = 1 / (mf_info[3] - mf_info[2])
        if in_val <= mf_info[0] or in_val >= mf_info[-1]:
            return 0.
        elif mf_info[1] <= in_val <= mf_info[2]:
            return 1.
        elif in_val < mf_info[1]:
            return delta_left * (in_val - mf_info[0])
        elif in_val > mf_info[2]:
            return delta_right * (mf_info[3] - in_val)
        return 0.

    return out_function


def get_membership_functions() -> dict:
    out_dict = {}
    for class_name in _CLASS_NAMES.keys():
        temp_dict = {}
        for product_name in _PRODUCT_NAMES.keys():
            temp_dict[product_name] = _get_membership_function(class_name, product_name)
        out_dict[class_name] = temp_dict
    return out_dict


def get_classes_list():
    return _CLASS_NAMES.keys()


def get_products_list():
    return _PRODUCT_NAMES.keys()


def get_total_weight(class_name: str) -> float:
    return _CLASS_TOTAL_WEIGHTS[class_name]


def get_weight(class_name: str, product_name: str) -> float:
    return _PRODUCT_WEIGHTS[product_name][class_name]


def get_possible_data_range(product_name: str) -> tuple:
    return _PRODUCT_DATA_RANGES[product_name]


def get_class_label(class_name: str) -> str:
    return _CLASS_LABELS[class_name]


def get_class_name(class_name: str) -> str:
    return _CLASS_NAMES[class_name]


def get_product_name_label(product_name: str) -> str:
    return _PRODUCT_NAMES[product_name]


def get_model_name(product_name: str) -> (str, str):
    return _MODEL_NAMES[product_name]


def is_class_supercooled(class_out: str) -> bool:
    return class_out in _SUPERCOOLED_WATER


def is_class_crystallized(class_out: str) -> bool:
    return class_out in _CRYSTALLIZED_WATER

