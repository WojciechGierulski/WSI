import random
from data import Nodes
import itertools


def process_nodes(nodes):
    bayes_net = {}
    for node in nodes:
        if len(node["dependencies"]) in bayes_net:
            bayes_net[len(node["dependencies"])].append(node)
        else:
            bayes_net[len(node["dependencies"])] = []
            bayes_net[len(node["dependencies"])].append(node)
    return bayes_net


def find_table(dependencies, prob_table):
    # dependencies - [bool, bool, ...]
    for i, table in enumerate(prob_table):
        if dependencies == table[0:-2]:
            return table, prob_table[i + 1]


def get_dependencies_values(dependencies, sample):
    result = []
    for name in dependencies:
        result.append(sample[name])
    return result


def generate_random_sample(bayes_net):
    sample = {}
    for dep_nr in range(len(list(bayes_net.keys()))):
        for node in bayes_net[dep_nr]:
            prob = random.uniform(0, 1)
            if dep_nr == 0:
                sample[node["name"]] = True if prob <= node["prob_table"][0][1] else False
            else:
                table1, table2 = find_table(get_dependencies_values(node["dependencies"], sample), node["prob_table"])
                prob = random.uniform(0, 1)
                sample[node["name"]] = True if prob <= table1[-1] else False
    return sample


def get_sample_prob(sample, nodes):
    prob = 1
    # sample - dict var: bool
    for node in nodes:
        if len(node["dependencies"]) != 0:
            depends = []
            for dependency in node["dependencies"]:
                depends.append(sample[dependency])
            table1, table2 = find_table(depends, node["prob_table"])
            if sample[node["name"]] is True:
                prob *= table1[-1]
            else:
                prob *= table2[-1]
        else:
            if sample[node["name"]] is True:
                prob *= node["prob_table"][0][1]
            else:
                prob *= node["prob_table"][1][1]
    return prob


def get_all_samples_prob(nodes, options=None):
    samples = []
    if options is None:
        options = ["Bus late", "Alarm on", "Overslept", "Late for school"]
    l = len(options)
    combinations = list(itertools.product([True, False], repeat=l))
    for combination in combinations:
        sample = {}
        for i, option in enumerate(options):
            sample[option] = combination[i]
        samples.append(sample)
    return samples, [get_sample_prob(sample, nodes) for sample in samples]


network = process_nodes(Nodes)
samples, probs = get_all_samples_prob(Nodes)


samples_nr = 10000
samples_counter = [0 for _ in range(len(probs))]
for _ in range(samples_nr):
    sample = generate_random_sample(network)
    samples_counter[samples.index(sample)] += 1


# Normalize
samples_counter = [x/samples_nr for x in samples_counter]

print(probs)
print(samples_counter)