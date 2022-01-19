# Zakładana jest poprawnośc danych wejściowych
Nodes = [
    {
        "name": "Alarm on", "dependencies": [], "prob_table":
        [
            [True, 0.95],
            [False, 0.05]
        ]
    },
    {
        "name": "Bus late", "dependencies": [], "prob_table":
        [
            [True, 0.2],
            [False, 0.8]
        ]
    },
    {
        "name": "Overslept", "dependencies": ["Alarm on"], "prob_table":
        [
            [True, True, 0.1],
            [True, False, 0.9],
            [False, True, 0.7],
            [False, False, 0.3]
        ]
    },
    {
        "name": "Late for school", "dependencies": ["Overslept", "Bus late"], "prob_table":
        [
            [True, True, True, 0.99],
            [True, True, False, 0.01],
            [True, False, True, 0.94],
            [True, False, False, 0.06],
            [False, True, True, 0.6],
            [False, True, False, 0.4],
            [False, False, True, 0.2],
            [False, False, False, 0.8]
        ]
    },
]