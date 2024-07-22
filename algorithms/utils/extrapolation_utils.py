from copy import deepcopy
from typing import Iterable, Tuple, Dict, List, Literal, Union
from src.abstractions import Model, Data
from time import strftime, localtime
import os, sys
import random
import pandas as pd
import json
import datasets
from src.text_writer import write_log
import warnings
from tqdm import tqdm
from sympy import binomial
import numpy as np


def extrapolate(
    id: str,
    current_timestep: int,
    timesteps_ahead: int,
    extrapolation_order: int,
    preference_history: List[Data],
) -> Data:
    """Extrapolate the preference data using the preference history, looking timesteps_ahead timesteps ahead."""

    exp_order = min(extrapolation_order, len(preference_history) - 1)
    warned = False

    constructed_data = []

    print(f"Extrapolating with order {exp_order} (full order {extrapolation_order})")
    num_missing = 0

    data_arrays = [
        list(preference_history[i].all_passages())
        for i in range(len(preference_history) - exp_order - 1, len(preference_history))
    ]

    assert all(
        len(data_arrays[i]) == len(data_arrays[0]) for i in range(len(data_arrays))
    ), "Data arrays have different lengths"
    final_coefs = []

    for sample_tuple in zip(*data_arrays):
        if not sample_tuple[-1]:
            num_missing += 1
            continue

        hash_vals = [
            hash(json.dumps(sample_tuple[i], sort_keys=True)) if sample_tuple[i] else 0
            for i in range(exp_order + 1)
        ]

        # There can only be up to two unique hash values in the tuple, representing the two preference orderings of the two candidate responses
        if len(set([val for val in hash_vals if val != 0])) > 2:
            print(
                f"Error: More than two unique hash values in the tuple {sample_tuple}"
            )
            raise ValueError("More than two unique hash values in the tuple")

        result_dict = deepcopy(sample_tuple[-1])
        assert (
            len(result_dict["output"]) == 2
        ), "There must be exactly two candidate responses"
        coef_array = np.array(
            [
                binomial(exp_order + 1, i) * (-1) ** (exp_order - i)
                for i in range(exp_order + 1)
            ],
            dtype=int,
        )
        vote_array = np.array(
            [
                0 if hash_vals[i] == 0 else 1 if hash_vals[i] == hash_vals[-1] else -1
                for i in range(exp_order + 1)
            ],
            dtype=int,
        )

        for _ in range(timesteps_ahead):
            vote_array[0] = np.dot(coef_array, vote_array)
            vote_array = np.roll(vote_array, -1)

        assert vote_array.shape == (exp_order + 1,), "Vote array has incorrect shape"
        if abs(vote_array[-1]) > 5 and not warned:
            warnings.warn(
                f"Extrapolation result is large ({vote_array[-1]} out of {vote_array}), and may not be reliable."
            )
            warned = True

        final_coefs.append(vote_array[-1])
        assert (
            vote_array[-1] != 0
        ), "Extrapolation result is zero, which should never happen if implementation is correct."
        if vote_array[-1] < 0:
            result_dict["output"] = list(reversed(result_dict["output"]))
            vote_array[-1] *= -1

        for _ in range(min(vote_array[-1], 5)):
            constructed_data.append(result_dict)

    final_coefs = np.array(final_coefs)
    print(
        f"Constructed {len(constructed_data)} preference data points; discarded {num_missing} due to missing responses."
    )
    print(f"Extrapolation mean coef: {np.mean(final_coefs)}")
    print(f"Extrapolation std coef: {np.std(final_coefs)}")
    print(f"Extrapolation max coef: {np.max(final_coefs)}")
    print(f"Extrapolation min coef: {np.min(final_coefs)}")
    print(f"Extrapolation mean abs coef: {np.mean(np.abs(final_coefs))}")
    # if num_missing >= max(25, len(constructed_data) / 2):
    #     raise ValueError(f'Too many missing responses ({num_missing}) in the extrapolation data (eventual size {len(constructed_data)}).')

    result = Data(
        data_name=f"{id}_{current_timestep}",
        data_type="preference",
        data_content=constructed_data,
    )

    result.key_fields = preference_history[-1].key_fields
    return result
