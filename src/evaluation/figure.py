import utils as ut
import quantify as qt
import json
import numpy as np


def get_dim(template_raw, template_out, idx):
    vec_set = []
    for i in idx:
        print("now", i)
        path1 = template_raw.format(n=str(i))
        path2 = template_out.format(n=str(i))
        with open(path1, "r") as f:
            raw = json.load(f)
        results = ut.collect_dim(raw)
        print("collected. length:", len(results["0"]))
        with open(path2, "w") as f:
            json.dump(results, f)
        vec_set.append(np.array(results["avg"]))
    return vec_set


template1 = (
    "output/evaluation_results/8b-C0{n}-instruct_single/8b-C0{n}-instructraw.json"
)
template2 = (
    "output/evaluation_results/8b-C0{n}-instruct_single/8b-C0{n}-instructdim.json"
)
idx = [13, 14, 15, 16, 17, 18, 19, 20, 21]
vecs = get_dim(template1, template2, idx)

"""
qt.plot_vectors(qt.normalize_by_sum([x[:5] for x in vecs]), idx[0], 'b')
qt.plot_vectors(qt.normalize_by_sum([x[5:10] for x in vecs]), idx[0], 's')
qt.plot_vectors(qt.normalize_by_sum([x[10:15] for x in vecs]), idx[0], 'f')
qt.plot_vectors(qt.normalize_by_sum([x[15:19] for x in vecs]), idx[0], 'v')

vecs = qt.standardize_vectors(vecs)
qt.analyze_vectors_quadratic(vecs)
"""
qt.plot_cosine_similarity_heatmap([x[:5] for x in vecs], "b")
qt.plot_cosine_similarity_heatmap([x[5:10] for x in vecs], "s")
qt.plot_cosine_similarity_heatmap([x[10:15] for x in vecs], "f")
qt.plot_cosine_similarity_heatmap([x[15:19] for x in vecs], "v")
