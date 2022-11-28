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
    fid.close()

with open('rain.bin', 'rb') as fid:
    raw_bytes = fid.read()
    data_r = np.frombuffer(raw_bytes, dtype=float)
    data_r = np.array(data_r.reshape([int(len(data_r) / 5), 5]))
    fid.close()

with open('drizzle.bin', 'rb') as fid:
    raw_bytes = fid.read()
    data_drizzle = np.frombuffer(raw_bytes, dtype=float)
    data_drizzle = data_drizzle.reshape([int(len(data_drizzle) / 5), 5])
    fid.close()


def calculate(data_vector: np.array, left_bound: float, right_bound: float) -> float:
    # for val in data_vector:
    #     if val < left_bound or val > right_bound:
    #         print(f'{left_bound} < {val} < {right_bound}')
    return ((left_bound < data_vector) & (data_vector < right_bound)).sum() / data_vector.size

# Drizzle - Rain - Dry snow - Ice Crystals - Wet snow
def print_product(name: str, MF_values: list, num_product: int):
    return
    print('--------------')
    probability_drizzle = calculate(data_drizzle[:, num_product], *MF_values[0])
    print(f'{name} for drizzle = {probability_drizzle}')
    probability_rain = calculate(data_r[:, num_product], *MF_values[1])
    print(f'{name} for rain = {probability_rain}')
    probability_ds = calculate(data_ds[:, num_product], *MF_values[2])
    print(f'{name} for dry snow = {probability_ds}')
    probability_ic = calculate(data_ic[:, num_product], *MF_values[3])
    print(f'{name} for ice crystals = {probability_ic}')
    probability_ws = calculate(data_ws[:, num_product], *MF_values[4])
    print(f'{name} for wet snow = {probability_ws}')
    probability_mean = (probability_drizzle + probability_rain + probability_ds + probability_ic + probability_ws) / 5
    print(f'{name} mean = {probability_mean}')

def print_integral(data_id, mf_values):
    if data_id == 0:
        data = data_drizzle
    elif data_id == 1:
        data = data_r
    elif data_id == 2:
        data = data_ds
    elif data_id == 3:
        data = data_ic
    elif data_id == 4:
        data = data_ws
    else:
        print('error')
        quit()

    counter = 0
    for vals in data:
        found_fault = False
        for step in [0, 1, 2, 4]:
            if not (mf_values[step][0] <= vals[step] <= mf_values[step][1]):
                found_fault = True
                # print(mf_values[step][0], vals[step], mf_values[step][1])
                break
        if not found_fault:
            counter += 1
    return counter / 1000
    # print(counter)


# Drizzle - Rain - Dry snow - Ice Crystals - Wet snow
MFs_Zh = [(-100, 25), (21, 62), (-100,  40), (-100, 31), (-100, 41)]
print_product('Zh', MFs_Zh, 0)

MFs_Zdr = [(-0.1, 1), (-0.1, 3), (-0.4,  0.8), (-0.1, 5), (-0.1, 3.5)]
print_product('Zdr', MFs_Zdr, 1)

MFs_Ldr = [(-100, -35), (-47, -22), (-100,  -33), (-45, -20), (-34, -11)]
print_product('Ldr', MFs_Ldr, 2)

MFs_ro = [(0.997, 1.1), (0.98, 1.1), (0.98,  1.1), (0.95, 1.1), (0.8, 0.96)]
print_product('Rohv', MFs_ro, 3)

MFs_kdp = [(-0.1, 0.1), (0, 100), (-0.9,  0.7), (0, 0.7), (-0.1, 2.8)]
print_product('kdp', MFs_kdp, 4)

print('--------------')
for step, name in zip(range(5), ['Морось', 'Дождь', 'Сухой снег', 'Кристаллы', 'Мокрый снег']):
    mfs = [MFs_Zh[step], MFs_Zdr[step], MFs_Ldr[step], MFs_ro[step], MFs_kdp[step]]
    print(f'Вероятность для {name} равна {print_integral(step, mfs)}')


