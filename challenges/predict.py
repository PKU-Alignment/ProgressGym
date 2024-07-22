from benchmark.framework import JudgeBase, ExamineeBase
from typing import Iterable, Tuple, Dict, Union, List
from benchmark import FollowJudge
from src.abstractions import Model, Data


class PredictJudge(FollowJudge):

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)

        assert "timesteps_ahead" in kwargs, "timesteps_ahead must be provided"
        assert (
            0
            < self.current_timestep + int(kwargs["timesteps_ahead"])
            < len(self.model_list)
        ), "timesteps_ahead must be within the range of the model list"
        self.timesteps_ahead: int = int(kwargs["timesteps_ahead"])
        self.model_ahead: Model = self.model_list[
            self.current_timestep + self.timesteps_ahead
        ]
        self.supplementary_data["timesteps_ahead"] = self.timesteps_ahead

    def eval_snapshot(self, examinee: ExamineeBase) -> None:
        super().eval_snapshot(examinee, ground_truth_model=self.model_ahead)

    def tick(self) -> None:
        super().tick()

        if self.current_timestep + self.timesteps_ahead >= len(self.model_list):
            raise IndexError(
                "Cannot move forward, as the timesteps_ahead is out of range."
            )

        self.model_ahead = self.model_list[self.current_timestep + self.timesteps_ahead]

    def query_from_examinee(
        self, prompt: Union[str, Data, List[Dict]]
    ) -> Union[str, Data, List[Dict]]:
        result: str = super().query_from_examinee(prompt)
        return result
