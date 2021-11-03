import math
import random
import time

import numpy as np
from matplotlib import pyplot as plt
plt.style.use("ggplot")
from genetic_ops import crossing, mutation

# Params
u = 20
sigma = 0.05
pm = 1
pc = 0.5
elite_size = 1
imax = 1000
tournament_size = 2

def bird_plt(x, y):
    return np.sin(x) * np.e ** ((1 - np.cos(y)) ** 2) + np.cos(y) * np.e ** ((1 - np.sin(x)) ** 2) + (
            x - y) ** 2

def bird(v):
    x = v[0]
    y = v[1]
    return math.sin(x) * math.e ** ((1 - math.cos(y)) ** 2) + math.cos(y) * math.e ** ((1 - math.sin(x)) ** 2) + (
            x - y) ** 2


def generate_P0(size, clones=False):
    if clones:
        x = random.uniform(-2 * math.pi, 2 * math.pi)
        y = random.uniform(-2 * math.pi, 2 * math.pi)
        return np.array([[x, y] for _ in range(size)])

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
        choices = [random.randrange(0, Pt.shape[0]) for _ in range(t_size)]
        t_guys = [Pt[choice,:] for choice in choices]
        t_rates = [rates[choice] for choice in choices]
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
    HISTORY_r = np.zeros((imax,))
    HISTORY_p = np.zeros((imax, u, 2))
    i = 0
    rates = rate_population(P0, q)
    min_guy, min_rate = find_best_guy(P0, rates)
    Pt = P0
    while (i < imax):
        HISTORY_p[i,:,:] = Pt
        Pr = reproduction(Pt.copy(), rates, u, tournament_size)
        Pm = genetic_operations(Pr, sigma, pc, pm)
        new_rates = np.array([q(guy) for guy in Pm])
        best_guy, best_rate = find_best_guy(Pm, new_rates)
        if best_rate <= min_rate:
            min_rate = best_rate
            min_guy = best_guy
        #print(best_guy, best_rate)
        HISTORY_r[i] = best_rate
        Pt = succesion(Pt, Pm, q, elite_size)
        rates = np.array([q(guy) for guy in Pt])
        i += 1
    return min_guy, min_rate, HISTORY_r, HISTORY_p



## Plots ##
# Sila mutacji
"""
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
axes = [ax1, ax4, ax2, ax5,ax3, ax6]

p = generate_P0(u)

for s, sigmaa in enumerate([0.01, 0.2, 1]):
    g, r, h_r, h_p = ea(bird, p, u, sigmaa, pc, pm, tournament_size, elite_size, imax)
    for it in h_p:
        axes[2*s].plot([x[0] for x in it], [x[1] for x in it], 'r.', zorder=1)
    axes[2*s+1].plot(h_r, '-.')
    axes[2*s].title.set_text(f'Siła mutacji={sigmaa}')

x = np.linspace(-2*math.pi, 2*math.pi, 3000)
y = np.linspace(-2*math.pi, 2*math.pi, 3000)
X, Y = np.meshgrid(x, y)
Z = bird_plt(X, Y)
axes[0].contour(X, Y, Z, colors='black')
axes[2].contour(X, Y, Z, colors='black')
axes[4].contour(X, Y, Z, colors='black')

plt.show()
"""

# rozmiar elity
"""
p = generate_P0(u)
for size in [1, 8, 18]:
    g, r, h_r, h_p = ea(bird, p, u, sigma, pc, pm, tournament_size, size, imax)
    plt.plot(h_r, '--')

plt.legend(['rozmiar elity=1', 'rozmiar elity=8', 'rozmiar elity=18'])
plt.show()
"""

# rozmiar populacji
"""
times = []
hr_s = []

for uu in [5, 10, 20, 50, 100, 200]:
    start = time.time()
    p = generate_P0(uu)
    g, r, h_r, h_p = ea(bird, p, uu, sigma, pc, pm, tournament_size, elite_size, imax)
    end = time.time()
    hr_s.append(h_r)
    times.append(end - start)
    print(end-start)

fig, (ax1, ax2) = plt.subplots(1, 2)
for h in hr_s:
    ax2.plot(h)
ax1.plot([5, 10, 20, 50, 100, 200], times, 'o-')
ax1.title.set_text('Czas wykonania, a liczność populacji')
ax2.title.set_text('Funkcja celu dla różnych liczności populacji')
ax2.legend(['u=5', 'u=10', 'u=20', 'u=50', 'u=100'])

plt.show()
"""

# populacja inicjalna

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
axes = [ax1, ax4, ax2, ax5,ax3, ax6]

p = generate_P0(u)

for s, pop in enumerate([generate_P0(u), generate_P0(u, True), generate_P0(u, True)]):
    if s == 2:
        sigma = 0.5
    g, r, h_r, h_p = ea(bird, pop, u, sigma, pc, pm, tournament_size, elite_size, imax)
    for it in h_p:
        axes[2*s].plot([x[0] for x in it], [x[1] for x in it], 'r.', zorder=1)
    axes[2*s+1].plot(h_r, '-.')
    if s == 0:
        axes[2*s].title.set_text(f'Losowa populacja inicjalna')
    elif s==1:
        axes[2*s].title.set_text(f'Klony w populacji inicjalnej\nmała siła mutacji')
    else:
        axes[2*s].title.set_text(f'Klony w populacji inicjalnej\nduża siła mutacji')


x = np.linspace(-2*math.pi, 2*math.pi, 3000)
y = np.linspace(-2*math.pi, 2*math.pi, 3000)
X, Y = np.meshgrid(x, y)
Z = bird_plt(X, Y)
axes[0].contour(X, Y, Z, colors='black')
axes[2].contour(X, Y, Z, colors='black')
axes[4].contour(X, Y, Z, colors='black')


plt.show()
