from ..abstractions.model import Model
from utils import generate_alpaca
import os, json
from multiprocessing import freeze_support
import quantify as qt

"""
generate_alpaca('mc', os.path.join('src', 'evaluation', 'raw_dataset', 'moralchoice'))
generate_alpaca('views', os.path.join('src', 'evaluation', 'raw_dataset', 'views'))
generate_alpaca('foundation', os.path.join('src', 'evaluation', 'raw_dataset', 'foundation'))
"""
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICE"] = "0,1,2,3,4,6"
    freeze_support()
    set_model = [
        "8b-C013-instruct",
        "8b-C014-instruct",
        "8b-C015-instruct",
        "8b-C016-instruct",
        "8b-C017-instruct",
        "8b-C018-instruct",
        "8b-C019-instruct",
        "8b-C020-instruct",
        "8b-C021-instruct",
    ]
    vec = []
    for m in set_model:
        boi = Model(m, num_gpus=4)
        v = boi.evaluate(method="fast")
        # v = qt.calculate_model('output/evaluation_results/' + m + '_single/', m)
        vec.append(v)
    test_name = "8b_13to21"
    with open("output/evaluation_results/" + test_name + ".json", "w") as f:
        lst = [list(boi) for boi in vec]
        json.dump(lst, f)
    # qt.plot_parallel_coordinates(vec)
    qt.plot_heatmap(vec)
