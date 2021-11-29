import math
import random
import copy

import numpy as np

"""
Legend
'x' is max player - symbol is 1
'o' is min player - symbol is -1
0 is empty cell
"""

searched = {1: 0, -1: 0}

def draw_board(state):
    def c(x):
        if x == 1:
            return 'x'
        elif x == -1:
            return  'o'
        else:
            return ' '
    print(f"{c(state[0])}|{c(state[1])}|{c(state[2])}\n")
    print(f"{c(state[3])}|{c(state[4])}|{c(state[5])}\n")
    print(f"{c(state[6])}|{c(state[7])}|{c(state[8])}\n")

def if_terminal_state(state):
    if state[0] == state[4] == state[8] != 0:
        return True
    if state[6] == state[4] == state[2] != 0:
        return True
    if state[0] == state[3] == state[6] != 0:
        return True
    if state[3] == state[4] == state[5] != 0:
        return True
    if state[0] == state[1] == state[2] != 0:
        return True
    if state[2] == state[5] == state[8] != 0:
        return True
    if state[6] == state[7] == state[8] != 0:
        return True
    if state[1] == state[4] == state[7] != 0:
        return True
    if 0 not in state:
        return True
    return False


def terminal_payoff(state):
    if state[0] == state[4] == state[8] != 0:
        return [25, state] if state[0] == 1 else [-25, state]
    if state[6] == state[4] == state[2] != 0:
        return [25, state] if state[6] == 1 else [-25, state]
    if state[0] == state[3] == state[6] != 0:
        return [25, state] if state[0] == 1 else [-25, state]
    if state[3] == state[4] == state[5] != 0:
        return [25, state] if state[3] == 1 else [-25, state]
    if state[0] == state[1] == state[2] != 0:
        return [25, state] if state[0] == 1 else [-25, state]
    if state[2] == state[5] == state[8] != 0:
        return [25, state] if state[2] == 1 else [-25, state]
    if state[6] == state[7] == state[8] != 0:
        return [25, state] if state[6] == 1 else [-25, state]
    if state[1] == state[4] == state[7] != 0:
        return [25, state] if state[1] == 1 else [-25, state]
    if 0 not in state:
        return [0, state]
    print("error")


def h(state):
    # Heuristic
    return [(state[0] * 3) + (state[1] * 2) + (state[2] * 3) + (state[3] * 2) + (state[4] * 4) + (state[5] * 2) + (
            state[6] * 3) + (state[7] * 2) + (state[8] * 3), state]


def successors(state, to_move):
    successors = []
    for i, cell_state in enumerate(state):
        if cell_state == 0:
            successor = np.copy(state)
            successor[i] = to_move
            successors.append(successor)
            continue
    return successors


def get_random_move(state, to_move):
    successors = []
    for i, cell_state in enumerate(state):
        if cell_state == 0:
            successor = np.copy(state)
            successor[i] = to_move
            successors.append(successor)
            continue
    return random.choice(successors)


def alpha_beta(state, d, alpha, beta, to_move, who_invoked):
    global searched
    if if_terminal_state(state):
        return terminal_payoff(state)
    elif d == 0:
        return h(state)
    U = successors(state, to_move)
    if to_move == 1:  # max move
        maxEval = -math.inf
        next_move = None
        for u in U:
            eval = alpha_beta(u, d - 1, alpha, beta, -1, who_invoked)[0]
            searched[who_invoked] += 1
            if eval > maxEval:
                next_move = u
                maxEval = eval
            alpha = max(alpha, eval)
            if alpha >= beta:
                break
        return [maxEval, next_move]
    elif to_move == -1:  # min to move
        minEval = math.inf
        next_move = None
        for u in U:
            eval = alpha_beta(u, d - 1, alpha, beta, 1, who_invoked)[0]
            searched[who_invoked] += 1
            if eval < minEval:
                minEval = eval
                next_move = u
            beta = min(beta, eval)
            if alpha >= beta:
                break
        return [minEval, next_move]


def minmax(state, d, to_move, who_invoked):
    global searched
    if if_terminal_state(state):
        return terminal_payoff(state)
    elif d == 0:
        return h(state)
    U = successors(state, to_move)
    next_move = None
    if to_move == 1:
        maxEval = -math.inf
        for u in U:
            eval = minmax(u, d - 1, to_move * -1, who_invoked)[0]
            searched[who_invoked] += 1
            if eval > maxEval:
                maxEval = eval
                next_move = u
        return [maxEval, next_move]
    elif to_move == -1:
        minEval = math.inf
        for u in U:
            eval = minmax(u, d - 1, to_move * -1, who_invoked)[0]
            searched[who_invoked] += 1
            if eval < minEval:
                minEval = eval
                next_move = u
        return [minEval, next_move]


class Player:
    def __init__(self, player, random, d, prune):
        # player is 1 for 'x' and -1 for 'o'
        self.player = player
        self.random = random
        self.d = d
        self.prune = prune

    def make_move(self, state):
        if self.random:
            return [0, get_random_move(state, self.player)]
        else:
            if self.prune:
                return alpha_beta(state, self.d, -math.inf, math.inf, self.player, self.player)
            else:
                return minmax(state, self.d, self.player, self.player)


player1 = Player(1, False, 1, True) # player, random, d, prune
player2 = Player(-1, False, 1, True)
current_state = np.array([0 for _ in range(9)])
current_player = player1
i = 0

while True:
    result = current_player.make_move(current_state)
    if if_terminal_state(current_state):
        break
    current_state = result[1]
    current_player = player1 if current_player == player2 else player2
    draw_board(current_state)
    print(result[0])
    i += 1

print(searched)
