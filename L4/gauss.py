import math
from statistics import NormalDist


class GaussFunc:
    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def calculate_value(self, x):
        exponent = math.exp(-((x - self.mean) ** 2 / (2 * self.sd ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * self.sd)) * exponent
