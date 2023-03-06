from scattering_simulation.calculation.meteo_products import *
from scattering_simulation.calculation.particle_properties.dielectric_permittivity.lossy_medium import *



x = WaterDropStatsModel(0.5, 1.4, 0, 5, 1e3, 2.1e4, -1, 4, )
# x = WaterDropStatsModel(1.4, 2., 0, 5, 1e3, 1e4, -1, 4, get_water_reflection_index(0.032, -5))
# x = WaterDropStatsModel(1.8, 3.6, 0, 5, 2e3, 9e4, -1, 4, get_water_reflection_index(0.032, 5))
# x = WaterDropStatsModel(1.8, 2.2, 0, 5, 2e3, 9e4, -1, 4, get_water_reflection_index(0.032, 5))
xx = CalculatorProducts(x)
for _ in range(30):
    xx.calculate(0.01, 0.4, 20)
    print('----')
