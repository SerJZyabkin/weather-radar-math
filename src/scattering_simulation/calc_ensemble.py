from scattering_simulation.calculation.tmatrix.ensemble import get_ensemble_generator

from scattering_simulation.statistical_model.create_histogram import HistogramDataAccumulator
from os import getcwd

path_to_hist_data = getcwd() + '/../../bin/hist_accum'

obj = HistogramDataAccumulator(path_to_hist_data, path_to_hist_data + "/config.json")
obj.execute('drizzle_compare', 11.5 * 0.01)