from src.abstractions.model import Model
from src.evaluation.utils import generate_alpaca
import os, json
from multiprocessing import freeze_support
import src.evaluation.quantify as qt

"""
generate_alpaca('mc', os.path.join('src', 'evaluation', 'raw_dataset', 'moralchoice'))
generate_alpaca('views', os.path.join('src', 'evaluation', 'raw_dataset', 'views'))
generate_alpaca('foundation', os.path.join('src', 'evaluation', 'raw_dataset', 'foundation'))
"""
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICE"] = "0"
    freeze_support()
    """
    set_model = ['8b-C013-instruct', '8b-C014-instruct','8b-C015-instruct', '8b-C016-instruct', '8b-C017-instruct', '8b-C018-instruct', '8b-C019-instruct', '8b-C020-instruct', '8b-C021-instruct']
    vec = []
    for m in set_model:
        boi = Model(m, num_gpus=8)
        v = boi.evaluate(method = 'fast')
        #v = qt.calculate_model('output/evaluation_results/' + m + '_single/', m)
        vec.append(v)
    test_name = '70b_13to21'
    with open('output/evaluation_results/' + test_name + '.json', 'w') as f:
        lst = [list(boi) for boi in vec]
        json.dump(lst, f)
    #qt.plot_parallel_coordinates(vec)
    vec = qt.standardize_vectors(vec)
    qt.analyze_vectors_quadratic(vec)
    """
    model = Model(
        "ExtrapolativeRLHFExaminee_05Jun155653_63908407_3_ExtrapolativeRLHFExaminee_05Jun155653_63908407_4",
        num_gpus=1,
    )
    vec = model.evaluate(method="fast")
    print(vec)
