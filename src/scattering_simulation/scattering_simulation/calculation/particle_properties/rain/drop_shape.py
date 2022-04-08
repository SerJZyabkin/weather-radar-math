from numpy import power


def get_gamma_rain(diameter_eqi: float):
    gamma = (0.9951 + 0.0251 * diameter_eqi - 0.03644 * power(diameter_eqi, 2.) + 0.005303 * power(diameter_eqi, 3.) -
             0.0002492 * power(diameter_eqi, 4.))
    return gamma


def get_shape(D):
    gamma = get_gamma_rain(D)
    a = D / 2 / power(gamma, 1 / 3)
    b = a * gamma
    return a, b