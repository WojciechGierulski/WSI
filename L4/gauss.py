import math


class GaussFunc:
    def __init__(self, mean, sd):
        self.var = float(sd) ** 2
        self.mean = mean
        self.denom = (2 * math.pi * self.var) ** 0.5

    def calculate_value(self, x):
        num = math.exp(-(float(x) - float(self.mean)) ** 2 / (2 * self.var))
        return num / self.denom
