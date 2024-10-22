from ..abstractions import Model
from .utils import generate_alpaca
import os, json
from multiprocessing import freeze_support
from . import quantify as qt

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
    vec = []
    for m in set_model:
        boi = Model(m)
        v = boi.evaluate(method="fast")
        # v = qt.calculate_model('output/evaluation_results/' + m + '_single/', m)
        vec.append(v)
    test_name = "8b_13to21"
    with open("output/evaluation_results/" + test_name + ".json", "w") as f:
        lst = [list(boi) for boi in vec]
        json.dump(lst, f)
    # qt.plot_parallel_coordinates(vec)
    qt.plot_heatmap(vec)
