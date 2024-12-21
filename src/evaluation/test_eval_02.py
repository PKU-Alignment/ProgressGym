from src.abstractions.model import Model
from src.evaluation.utils import generate_alpaca, collect
import os, json
from multiprocessing import freeze_support
import src.evaluation.quantify as qt
import numpy as np
from src.path import root

"""
generate_alpaca('mc', os.path.join(root, 'src', 'evaluation', 'raw_dataset', 'moralchoice'))
generate_alpaca('views', os.path.join(root, 'src', 'evaluation', 'raw_dataset', 'views'))
generate_alpaca('foundation', os.path.join(root, 'src', 'evaluation', 'raw_dataset', 'foundation'))
"""
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
        "8B-C021-instruct",
    ]
    with open("src/evaluation/assets/input_alpaca.json", "r") as f:
        ref = json.load(f)
    display = []
    for m in set_model:
        with open(
            f"{root}/output/datasets/evaluation_output_mc_" + m + ".json", "r"
        ) as f:
            d = json.load(f)
        raw = collect(d, logprobs=True)
        with open(
            f"{root}/output/evaluation_results/" + m + "_single/" + m + "_raw.json", "w"
        ) as f:
            json.dump(raw, f)
        v = qt.calculate_model(f"{root}/output/evaluation_results/" + m + "_single/", m)
        vec.append(v)
    test_name = "8b_all_fixed"
    vec = np.array(vec)
    with open(f"{root}/output/evaluation_results/" + test_name + ".json", "w") as f:
        lst = [list(boi) for boi in vec]
        json.dump(lst, f)
    qt.plot_heatmap(vec[:, 10:15], test_name + "_foundation", label_set=2, norm="group")
    qt.plot_heatmap(vec[:, 15:19], test_name + "_view", label_set=3, norm="group")
    qt.plot_heatmap(vec[:, :10], test_name + "_morality", label_set=1, norm="group")
