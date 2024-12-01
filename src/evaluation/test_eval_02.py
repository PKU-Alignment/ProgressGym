from src.abstractions.model import Model
from src.evaluation.utils import generate_alpaca, collect
import os, json
from multiprocessing import freeze_support
import src.evaluation.quantify as qt
import numpy as np
import random

if __name__ == "__main__":
    freeze_support()
    set_model = [
        "8B-C013-instruct",
        "8B-C014-instruct",
        "8B-C015-instruct",
        "8B-C016-instruct",
        "8B-C017-instruct",
        "8B-C018-instruct",
        "8B-C019-instruct",
        "8B-C020-instruct",
        "8B-C021-instruct"
    ]
    with open('src/evaluation/assets/input_alpaca.json', 'r') as f:
        ref = json.load(f)
    display = []
    for m in set_model:
        with open('output/evaluation_results/' + m + '_single/' + m + '_raw.json', 'r') as f:
            data = json.load(f)
        inputs = random.sample(ref, 3)
        for input in inputs:
            s, q, t, map, predicts = input['scenario_id'], input['question_type'], input['input'], input["mapping"], input["predict"]
            #probs = list(data[s][q][:-1] / data[s][q][-1])
            probs = [x / data[s][q][-1] for x in data[s][q][:-1]]
            probs = [probs[i-1] for i in map]
            display.append({"model": m, "question": t, "probs": probs})
    with open('output/evaluation_results/display.json', 'w') as f:
        json.dump(display, f)
     
        