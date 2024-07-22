import os, json
import numpy as np


def cosine_similarity(v1, v2):
    if np.linalg.norm(v1) * np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def sum_of_maxes(vectors):
    total_sum = 0
    n = len(vectors)

    for i in range(n):
        max_value = max(vectors[i:])
        total_sum += max_value

    return total_sum


def get_simple_diff_seq(actual, predict):
    ret = []
    for i in range(len(predict)):
        ret.append(
            cosine_similarity(actual[i + len(actual) - len(predict)], predict[i])
        )
    return ret


def get_diff_seq(actual, predict):
    ret = []
    for i in range(len(predict)):
        ans = 0
        for j in range(len(predict) - i):
            vec_actual = actual[i + len(actual) - len(predict) + j]
            ans += cosine_similarity(vec_actual, predict[i])
        ret.append(ans)
    return ret


def merge_dicts(*dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result


def get_score_predict(file):
    file_name = file.split("/")[-1].split(".")[0]
    with open(file, "r") as f:
        boi = json.load(f)
    ret = {}
    for key in boi.keys():
        if "Follow" not in boi[key].keys():
            print("No Follow for", key)
            continue
        if "supplementary_data" not in boi[key]["Follow"].keys():
            print("not loading", file_name, key)
            continue
        actual = boi[key]["Follow"]["supplementary_data"]["actual_vector"]
        # predict = boi[key]["Follow"]["supplementary_data"]["examinee_vector"]
        predict = actual
        ret[file_name + "_" + key] = sum_of_maxes(get_diff_seq(actual, predict))
    return ret


def get_score_coevolve(file):
    file_name = file.split("/")[-1].split(".")[0]
    with open(file, "r") as f:
        boi = json.load(f)
    ret = {}
    for key in boi.keys():
        if "Coevolve" not in boi[key].keys():
            print("No Coevolve for", file_name)
            continue
        if "supplementary_data" not in boi[key]["Coevolve"].keys():
            print("not loading", file_name, key)
            continue
        actual = boi[key]["Coevolve"]["supplementary_data"]["actual_model_vector"]
        # predict = boi[key]["Coevolve"]["supplementary_data"]["simulated_model_vector"]
        predict = actual
        ret[file_name + "_" + key] = sum_of_maxes(get_diff_seq(actual, predict))
    return ret


def get_score_follow(file):
    file_name = file.split("/")[-1].split(".")[0]
    with open(file, "r") as f:
        boi = json.load(f)
    ret = {}
    for key in boi.keys():
        if "Follow" not in boi[key].keys():
            print("No Follow for", file_name)
            continue
        if "supplementary_data" not in boi[key]["Follow"].keys():
            print("not loading", file_name, key)
            continue
        actual = boi[key]["Follow"]["supplementary_data"]["actual_vector"]
        predict = boi[key]["Follow"]["supplementary_data"]["examinee_vector"]
        ret[file_name + "_" + key] = sum(get_simple_diff_seq(actual, predict))
    return ret


def predict(path):
    result = {}
    for p in os.listdir(path):
        if (
            p.startswith("initial")
            or p.startswith("predict")
            or p.startswith("coevolve")
        ):
            continue
        score = get_score_predict(os.path.join(path, p))
        result.update(score)
    with open("output/benchmark_results/predict.json", "w") as f:
        json.dump(result, f)


def coevolve(path):
    result = {}
    for p in os.listdir(path):
        if (
            p.startswith("initial")
            or p.startswith("predict.")
            or p.startswith("coevolve.")
        ):
            continue
        score = get_score_coevolve(os.path.join(path, p))
        result.update(score)
    with open("output/benchmark_results/coevolve.json", "w") as f:
        json.dump(result, f)


def follow(path):
    result = {}
    for p in os.listdir(path):
        if (
            p.startswith("initial")
            or p.startswith("predict.")
            or p.startswith("coevolve.")
        ):
            continue
        score = get_score_follow(os.path.join(path, p))
        result.update(score)
    with open("output/benchmark_results/follow.json", "w") as f:
        json.dump(result, f)


# predict('output/benchmark_results/complete_results')
# coevolve('output/benchmark_results/complete_results')
# follow('output/benchmark_results/complete_results')


def calculate_score(key, dict):
    if key == "Follow":
        actual = dict["supplementary_data"]["actual_vector"]
        predict = dict["supplementary_data"]["examinee_vector"]
        dict["score"] = sum(get_simple_diff_seq(actual, predict))
        return dict
    elif key == "Predict":
        actual = dict["supplementary_data"]["actual_vector"]
        predict = dict["supplementary_data"]["examinee_vector"]
        dict["score"] = sum_of_maxes(get_diff_seq(actual, predict))
    elif key == "Coevolve":
        actual = dict["supplementary_data"]["actual_model_vector"]
        predict = dict["supplementary_data"]["simulated_model_vector"]
        dict["score"] = sum_of_maxes(get_diff_seq(actual, predict))
    return dict


def append_score(path):
    with open(path, "r") as f:
        result = json.load(f)
    for e_key in result.keys():
        for t_key in result[e_key].keys():
            task_result = result[e_key][t_key]
            result[e_key][t_key] = calculate_score(t_key, task_result)
    with open(path, "w") as f:
        json.dump(result, f)
