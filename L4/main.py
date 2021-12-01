import pandas as pd
from gauss import GaussFunc
import math

data = pd.read_csv("winequality-red.csv", sep=";")


class Model:
    def __init__(self, funcs, train_set):
        self.gauss_funcs = funcs
        self.qualities_counts = {}
        for quality in range(3, 9):
            self.qualities_counts[quality] = train_set['quality'].value_counts()[quality]
        self.records_count = sum([value for key, value in self.qualities_counts.items()])

    def predict(self, row):
        pass


def divide_train_test(data):
    shuffled = data.sample(frac=1, random_state=200)
    train_size = int(0.6 * len(data))
    train_set = shuffled[:train_size]
    test_set = shuffled[train_size:]
    return train_set, test_set


def divide_cross_validation(data, k=5):
    shuffled = data.sample(frac=1, random_state=200)
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


train, test = divide_train_test(data)
model = train_model(train)
score = test_model(model, test)
