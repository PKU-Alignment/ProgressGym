import os, json
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from ..abstractions import Model
from .utils import generate_alpaca, _collect
from multiprocessing import freeze_support
from . import quantify as qt
import numpy as np
"""
generate_alpaca('mc', os.path.join('src', 'evaluation', 'raw_dataset', 'moralchoice'))
generate_alpaca('views', os.path.join('src', 'evaluation', 'raw_dataset', 'views'))
generate_alpaca('foundation', os.path.join('src', 'evaluation', 'raw_dataset', 'foundation'))
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
    set_model = set_model[:2]
    vec = []
    for m in set_model:
        boi = Model(m)
        v = boi.evaluate(method="fast", logprobs = True)
        '''
        with open("output/datasets/evaluation_output_mc_" + m + ".json", 'r') as f:
            d = json.load(f)
        raw = _collect(d)
        with open('output/evaluation_results/' + m + '_single/' + m + '_raw.json', 'w') as f:
            json.dump(raw, f)
        v = qt.calculate_model('output/evaluation_results/' + m + '_single/', m)
        '''
        vec.append(v)
    test_name = "logprob_test"
    with open("output/evaluation_results/" + test_name + ".json", "w") as f:
        lst = [list(boi) for boi in vec]
        json.dump(lst, f)
    vec = np.array(vec)
    # qt.analyze_vectors_quadratic(vec)
    # vec = json.load(open("output/evaluation_results/" + test_name + ".json", "r"))
    # qt.plot_parallel_coordinates(vec)
    qt.plot_heatmap(vec[:, 10:15], test_name + '_foundation', label_set = 2, norm = "group")
    qt.plot_heatmap(vec[:, 15:19],  test_name + '_view',label_set = 3, norm = "group")
    qt.plot_heatmap(vec[:, :10], test_name + '_morality', label_set = 1, norm = "group")
