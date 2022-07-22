import numpy
# !/usr/bin/env python3
import os
from scattering_simulation.calculation.common_conversions import *
from scattering_simulation.calculation.particle_properties.dielectric_permittivity.lossy_medium import *
from scattering_simulation.calculation.particle_properties.dielectric_permittivity.medium_mixture import *
from scattering_simulation.calculation.particle_properties.rain.drop_size import *
from scattering_simulation.calculation.particle_properties.rain.drop_shape import *
from scattering_simulation.calculation.particle_properties.rain.physical_parameters import *
import matplotlib.pyplot as plt
from scattering_simulation.calculation.rayleigh_approximation.spheroid_scattering import *
import numpy as np
from scattering_simulation.calculation.tmatrix.spheroid import *
from scattering_simulation.calculation.tmatrix.lpd import *

# wave_length = 0.032
# temperature = -10
#
# eps_ice = get_ice_dielectric_constant(wave_length, temperature)
# eps_air = get_air_dielectric_constant(wave_length)
# fractions_air = np.arange(0, 1, 0.001)
# fractions_ice = 1 - fractions_air
#
# vals = get_mixing_maxwell_garnett(eps_air, eps_ice, fractions_air, fractions_ice)
# plt.plot(fractions_ice, vals.real, 'k-.')
#
# vals = get_mixing_maxwell_garnett(eps_ice, eps_air, fractions_ice, fractions_air)
# plt.plot(fractions_ice, vals.real, 'k--')
#
# vals = np.array([get_mixing_ponder_vansandern(eps_air, eps_ice, f1, f2) for f1, f2 in zip(fractions_air, fractions_ice)])
# plt.plot(fractions_ice, vals.real, 'k')
#
# plt.show()
import scattering_simulation.calculation.tmatrix.ampld_lp_converted as lib

def plot_distrubution():
    distr_mr = factory_normalized_gamma_distribution(1e4, 2.1, 1.5)
    distr_lr = factory_exponential_distribution(92, 1.06)# factory_normalized_gamma_distribution(280, 0, 1.06)
    distr_bd = factory_normalized_gamma_distribution(300, 0.7, 3.3)
    #
    particle_size = np.arange(0, 9, 0.01)
    #
    plt.plot(particle_size, distr_mr(particle_size))

    plt.show()

    print(total_number_consentration(distr_lr, 0, 16, 0.01))
    print(rain_water_content(distr_lr, 0, 16, 0.01))

def plot_relei_spheroid():
    temperature = 10
    wavelength = 0.15

    diel_k = get_water_dielectric_constant(wavelength, temperature)

    D = np.arange(0.1, 9, 0.1)
    s1 = np.zeros(D.shape)
    s2 = np.zeros(D.shape)

    for i in range(len(D)):
        a, b = get_shape(D[i])
        print(a, b, get_gamma_rain(D[i]))
        s1[i], s2[i] = np.abs(backscattering_amplitude(a , b , wavelength * 1000, diel_k))


    plt.plot(D, s1, '-k')
    plt.plot(D, s2, '--k')
    plt.ylim([1.e-6, 1])
    plt.yscale('log')
    plt.xlim([0, 8])
    plt.grid(True)
    plt.show()

#
# run_lpq()

X, W = lib.gauss(6, 0, True)
plot_relei_spheroid()
# #
# for x, w in zip(X, W):
#     print(x, w)