import numpy as np


def crossing(guy1, guy2, alpha=0.1):
    return guy1 * alpha + (1 - alpha) * guy2


def mutation(guy, sigma):
    return guy + sigma * np.array([np.random.normal() for _ in range(guy.shape[0])])
