from typing import Iterable, Tuple, Dict, List, Literal, Union
from src.abstractions import Model, Data
from time import strftime, localtime
import os, sys
import random
import pandas as pd
import json
import datasets
from src.text_writer import write_log
from benchmark import JudgeBase, ExamineeBase, PredictJudge
from algorithms.utils.rw_utils import elicit_rw_preference, default_rw_data
from algorithms.utils.extrapolation_utils import extrapolate
import warnings
from tqdm import tqdm
import numpy as np


class ExtrapolativeDPOExaminee(ExamineeBase):
    preference_history: List[Data]
    independent: bool

    def reset(self, **kwargs):
        super().reset(**kwargs)

        assert "timesteps_ahead" in kwargs, "timesteps_ahead must be provided"
        self.timesteps_ahead: int = int(kwargs["timesteps_ahead"])

        assert "extrapolation_order" in kwargs, "extrapolation_order must be provided"
        self.extrapolation_order: int = int(kwargs["extrapolation_order"])

        self.independent = "independent" in kwargs and eval(kwargs["independent"])

        self.rw_data: Union[Data, datasets.Dataset] = (
            default_rw_data.copy()
            if "preference_data" not in kwargs
            else kwargs["preference_data"]
        )
        self.preference_history = []

    def query_from_judge(
        self, prompt: Union[str, Data, List[Dict]]
    ) -> Union[str, Data, List[Dict]]:
        result: str = super().query_from_judge(prompt)
        return result

    def get_current_model(self) -> Model:
        result: Model = super().get_current_model()
        return result

    def run(self, judge: JudgeBase) -> Iterable:

        # Do some initializations of self.current_model by calling base class implementation
        super().run(judge)
        initial_model = self.current_model
        judge.supplementary_data["extrapolated_preference"] = []

        while True:
            print("Running DPO at timestep ", judge.current_timestep)

            try:
                oneoff_rw_data = Data(
                    f"preference_{self.checkpoint_id}_{judge.checkpoint_id}_{self.current_timestep}",
                    data_type="preference",
                )
                oneoff_rw_data.set_key_fields(
                    prompt_field_name="instruction",
                    query_field_name="input",
                    response_field_name="output",
                )
                print(f"Loaded preference data {oneoff_rw_data.data_path}.")
            except:
                oneoff_rw_data = elicit_rw_preference(self, judge, "deepspeed", True)

            self.preference_history.append(oneoff_rw_data)
            judge.supplementary_data["preference_history"] = [
                data.data_path for data in self.preference_history
            ]

            try:
                rw_data = Data(
                    f"{self.checkpoint_id}_{judge.checkpoint_id}_{judge.current_timestep}",
                    data_type="preference",
                )
                rw_data.key_fields = self.preference_history[-1].key_fields
                print(f"Loaded extrapolation data {rw_data.data_path}.")
            except:
                rw_data = extrapolate(
                    f"{self.instance_id}_{judge.instance_id}",
                    judge.current_timestep,
                    self.timesteps_ahead,
                    self.extrapolation_order,
                    self.preference_history,
                )
            judge.supplementary_data["extrapolated_preference"].append(
                rw_data.data_path
            )

            if self.independent:
                self.current_model = initial_model

            try:
                model = Model(
                    f"{self.checkpoint_id}_{judge.current_timestep}",
                    template_type=self.current_model.template_type,
                    num_gpus=self.current_model.num_gpus,
                )
                self.current_model = model
                print(f"Loaded checkpoint model {model.model_path}.")
            except:
                self.current_model = self.current_model.finetune(
                    data=rw_data,
                    result_model_name=f"{self.instance_id}_{judge.current_timestep}",
                    stage="dpo",
                    algo="full_param",
                )

            self.current_timestep += 1
            print("timestamp complete: ", self.current_timestep)

            yield self.current_timestep
