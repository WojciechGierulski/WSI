import random
import numpy as np
import copy
import matplotlib.pyplot as plt
plt.style.use("ggplot")

actions = {"RIGHT", "LEFT", "UP", "DOWN"}
epsilon = 0.1
learn_rate = 0.2
gamma = 0.8
cols = 13
rows = 9
max_it = 200
epochs = 500


def load_maze(cols=cols, rows=rows, path="maze.txt"):
    maze = np.empty((rows, cols))
    start_pos = None
    finish_pos = None
    free_cells = set()
    visited_cells = set()
    with open(path) as file:
        for row, line in enumerate(file):
            for col, char in enumerate(line):
                if char == ".":
                    maze[row, col] = 1
                    free_cells.add((row, col))
                elif char == "#":
                    maze[row, col] = 0
                elif char == "S":
                    start_pos = (row, col)
                    maze[row, col] = 1
                    free_cells.add((row, col))
                    visited_cells.add((row, col))
                elif char == "F":
                    finish_pos = (row, col)
                    maze[row, col] = 1
                    free_cells.add((row, col))
    return start_pos, finish_pos, maze, free_cells, visited_cells


def create_q_table(rows, cols, actions):
    Q = {}
    for row in range(rows):
        for col in range(cols):
            for action in actions:
                Q[((row, col), action)] = 0
    return Q


def generate_u(Q, x, epsilon, actions):
    nr = random.uniform(0, 1)
    if nr <= epsilon:
        return random.choice(list(actions))
    else:
        us = []
        values = []
        maxx_q = -float("inf")
        maxx_u = None
        for action in actions:
            us.append(action)
            values.append(Q[(x, action)])
        for u, value in zip(us, values):
            if value > maxx_q:
                maxx_q = value
                maxx_u = u
        return maxx_u


def make_u_and_get_reward(agent_position, u, free_cells, visited_cells, finish_pos):
    next_position = None
    reward = 0
    # Make action
    if u == "RIGHT":
        next_position = (agent_position[0], agent_position[1] + 1)
    elif u == "LEFT":
        next_position = (agent_position[0], agent_position[1] - 1)
    elif u == "UP":
        next_position = (agent_position[0] - 1, agent_position[1])
    elif u == "DOWN":
        next_position = (agent_position[0] + 1, agent_position[1])

    if next_position not in free_cells:  # Hit the wall edge
        next_position = agent_position
        return -0.5, next_position
    if next_position in visited_cells:
        return -1, next_position
    if next_position == finish_pos:  # maze finished
        return 5, next_position

    return -0.1, next_position  # for time spent in maze


def update_q_table(Q, u, x, next_x, learn_rate, reward, actions, gamma):
    maxx = -float("inf")
    for action in actions:
        if Q[(next_x, action)] > maxx:
            maxx = Q[(next_x, action)]
    Q[(x, u)] = Q[(x, u)] + learn_rate * (reward + gamma * maxx - Q[(x, u)])

def get_path(Q, start_pos, finish_pos):
    path = []
    actionss = []
    actual_pos = start_pos
    it = 1
    while it <= 30:
        path.append(actual_pos)
        options = [[action, Q[(actual_pos, action)]] for action in actions]
        maxx = max(options, key=lambda x: x[1])
        action = maxx[0]
        if action == "RIGHT":
            actual_pos = (actual_pos[0], actual_pos[1] + 1)
        elif action == "LEFT":
            actual_pos = (actual_pos[0], actual_pos[1] - 1)
        elif action == "UP":
            actual_pos = (actual_pos[0] - 1, actual_pos[1])
        elif action == "DOWN":
            actual_pos = (actual_pos[0] + 1, actual_pos[1])
        actionss.append(action)
        it += 1
        if actual_pos == finish_pos:
            print("found path")
            break
    return path, actionss


# Run Ql
start_pos, finish_pos, maze, free_cells, visited_cells_original = load_maze()
Q = create_q_table(rows, cols, actions)
total_rewards = []
total_steps = []
find_counter = 0

for epoch in range(epochs):
    print(epoch)
    it = 1
    total_reward = 0
    steps = 0
    agent_pos = start_pos
    visited_cells = copy.deepcopy(visited_cells_original)
    while True:
        next_u = generate_u(Q, agent_pos, epsilon, actions)
        reward, next_pos = make_u_and_get_reward(agent_pos, next_u, free_cells, visited_cells, finish_pos)
        visited_cells.add(next_pos)
        update_q_table(Q, next_u, agent_pos, next_pos, learn_rate, reward, actions, gamma)
        total_reward += reward
        steps += 1
        if reward == 5 or it == max_it:
            find_counter += 1 if reward == 5 else 0
            # koniec epizodu
            total_rewards.append(total_reward)
            total_steps.append(steps)
            break
        agent_pos = next_pos
        it += 1

path, actions = get_path(Q, start_pos, finish_pos)
print(path)
print(actions)
print(find_counter)


# Plots
plt.plot(total_rewards, ".")
plt.xlabel("Epoka")
plt.ylabel("Sumaryczna nagroda")
plt.figure()
plt.plot(total_steps, ".")
plt.xlabel("Epoka")
plt.ylabel("Liczba kroków")
plt.show()
print(Q)
