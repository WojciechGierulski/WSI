import math

import pandas as pd
from gauss import GaussFunc

data = pd.read_csv("winequality-red.csv", sep=";")


class Model:
    def __init__(self, funcs, train_set):
        self.gauss_funcs = funcs
        self.init_guesses = {}
        for quality in range(3, 9):
            self.init_guesses[quality] = train_set['quality'].value_counts()[quality] / len(train_set)

    def predict(self, row):
        qualities_predictions = {}
        for quality in range(3, 9):
            score = math.log(self.init_guesses[quality])
            for index, value in row.iteritems():
                if index != 'quality':
                    probability = self.gauss_funcs[index][quality].calculate_value(value)
                    if probability == 0.0:
                        probability = 1e-300
                    score += math.log(probability)
            qualities_predictions[quality] = score
        return max(qualities_predictions, key=qualities_predictions.get)


def divide_train_test(data):
    shuffled = data.sample(frac=1, random_state=101)
    train_size = int(0.6 * len(data))
    train_set = shuffled[:train_size]
    test_set = shuffled[train_size:]
    return train_set, test_set


def divide_cross_validation(data, k=5):
    shuffled = data.sample(frac=1, random_state=101)
    train_sets = []
    test_sets = []
    set_size = int(1 / k * len(data))
    all_sets = []
    for i in range(k):
        all_sets.append(shuffled[i * set_size:(i + 1) * set_size])
    for i in range(k):
        numbers = set([x for x in range(k)])
        numbers.remove(i)
        test_set = all_sets[i]
        train_set = pd.concat([all_sets[a] for a in numbers])
        train_sets.append(train_set)
        test_sets.append(test_set)
    return train_sets, test_sets


def train_model(train_set):
    train_by_quality = {}
    for quality in range(3, 9):
        train_by_quality[quality] = train_set[train_set['quality'] == quality]
    gauss_functions = {}
    for column_name in train_set.columns[0:-1]:
        gauss_functions[column_name] = {}
        for quality in range(3, 9):
            mean = train_by_quality[quality][column_name].mean()
            std = train_by_quality[quality][column_name].std()
            gauss_functions[column_name][quality] = GaussFunc(mean, std)
    return Model(gauss_functions, train_set)


def test_model(model, test):
    correct = 0
    incorrect = 0
    for index, row in test.iterrows():
        predicted_quality = model.predict(row)
        if predicted_quality == row['quality']:
            correct += 1
        else:
            incorrect += 1
    return correct / (correct + incorrect)


method = 'train_test'

if method == 'train_test':
    train, test = divide_train_test(data)
    model = train_model(train)
    score = test_model(model, test)
    print(score)
elif method == 'cross':
    k = 5
    train_sets, test_sets = divide_cross_validation(data, k)
    for it in range(k):
        model = train_model(train_sets[it])
        score = test_model(model, test_sets[it])
        print(score)

