import math
import random

import numpy as np
from genetic_ops import crossing, mutation

# Params
u = 20
sigma = 0.1
pm = 1
pc = 0.5
elite_size = 1
imax = 5000
tournament_size = 2


def bird(v):
    x = v[0]
    y = v[1]
    return math.sin(x) * math.e ** ((1 - math.cos(y)) ** 2) + math.cos(y) * math.e ** ((1 - math.sin(x)) ** 2) + (
            x - y) ** 2


def generate_P0(size):
    P = np.zeros((size, 2))
    for i in range(size):
        x = random.uniform(-2 * math.pi, 2 * math.pi)
        y = random.uniform(-2 * math.pi, 2 * math.pi)
        P[i, :] = np.array([x, y])
    return P


def rate_population(P, q):
    rates = np.zeros(P.shape[0])
    for i, guy in enumerate(P):
        rates[i] = q(guy)
    return rates


def find_best_guy(P, rates):
    min_rate = float("inf")
    min_guy = None
    for guy, rate in zip(P, rates):
        if rate < min_rate:
            min_rate = rate
            min_guy = guy
    return min_guy, min_rate


def reproduction(Pt, rates, u, t_size):
    # Tournament reproduction
    new_guys = np.zeros((u, 2))
    for i in range(u):
        t_guys = [random.choice(Pt) for _ in range(t_size)]
        t_rates = [rates[(np.where(t_guys == t_guy))[0][0]] for t_guy in t_guys]
        new_guys[i, :] = t_guys[t_rates.index(min(t_rates))]
    return new_guys


def genetic_operations(P, sigma, pc, pm):
    # Crossing
    new_guys = np.zeros((P.shape[0], 2))
    for i, guy in enumerate(P):
        if random.uniform(0, 1) <= pc:
            partner = random.choice(P)
            new_guy = crossing(guy, partner)
            new_guys[i, :] = new_guy
        else:
            new_guys[i, :] = guy

    # Mutations
    new_guys_2 = np.zeros((P.shape[0], 2))
    for i, guy in enumerate(new_guys):
        if random.uniform(0, 1) <= pm:
            new_guys_2[i, :] = mutation(guy, sigma)
        else:
            new_guys_2[i, :] = guy

    return new_guys_2


def succesion(Pt, Pm, q, elite_size):
    # Elite succession
    if elite_size:
        new_population = np.zeros((Pm.shape[0], 2))
        Pt = list(Pt)
        Pt.sort(key=lambda x: q(x))
        Pm = list(Pm)
        Pm.sort(key=lambda x: q(x))
        Pm = np.array(Pm)
        Pt = np.array(Pt)
        new_population[0:elite_size, :] = Pt[0:elite_size, :]
        new_population[elite_size:, :] = Pm[0:-elite_size, :]
        return new_population
    else:
        return Pm

def ea(q, P0, u, sigma, pc, pm, tournament_size, elite_size, imax):
    i = 0
    rates = rate_population(P0, q)
    min_guy, min_rate = find_best_guy(P0, rates)
    Pt = P0
    while (i < imax):
        Pr = reproduction(Pt.copy(), rates, u, tournament_size)
        Pm = genetic_operations(Pr, sigma, pc, pm)
        new_rates = np.array([q(guy) for guy in Pm])
        best_guy, best_rate = find_best_guy(Pm, new_rates)
        if best_rate <= min_rate:
            min_rate = best_rate
            min_guy = best_guy
        print(best_guy, best_rate)
        Pt = succesion(Pt, Pm, q, elite_size)
        rates = np.array([q(guy) for guy in Pt])
        i += 1
    return min_guy, min_rate


print(ea(bird, generate_P0(10 * u), u, sigma, pc, pm, tournament_size, elite_size, imax))
