from scattering_simulation.calculation.tmatrix.ampld_lp import calculate
import numpy as np
from scattering_simulation.calculation.particle_properties.dielectric_permittivity.lossy_medium import \
    get_water_reflection_index
from scattering_simulation.calculation.particle_properties.rain.drop_shape import get_gamma_rain
import matplotlib.pyplot as plt


wavelength = 32
temperature = 0

epsilon = get_water_reflection_index(wavelength * 10e-3, temperature)
print(epsilon)

Deff = 1.6
S = calculate(wavelength, Deff, epsilon, get_gamma_rain(Deff), 0, 90, 0)
print(20 * np.log10(np.abs(S[0] / S[3])), get_gamma_rain(Deff))

Deff = 1.8
S = calculate(wavelength, Deff, epsilon, get_gamma_rain(Deff), 0, 90, 0)
print(20 * np.log10(np.abs(S[0] / S[3])), get_gamma_rain(Deff))

S = calculate(wavelength, Deff, epsilon, 0.8, 0, 90, 0)
print(20 * np.log10(np.abs(S[0] / S[3])), get_gamma_rain(Deff))

Deff =  2
S = calculate(wavelength, Deff, epsilon, get_gamma_rain(Deff), 0, 90, 0)
print(20 * np.log10(np.abs(S[0] / S[3])), get_gamma_rain(Deff))
