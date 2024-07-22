from benchmark.framework import JudgeBase, ExamineeBase
from typing import Iterable, Tuple, Dict, Union, List
from src.abstractions import Model, Data


class DummyJudge(JudgeBase):
    """DummyJudge is a dummy judge that evaluates the performance of a dummy examinee in a trivial manner.
    It is only used for testing purposes. Do not use this judge for real benchmarking.
    """

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)

    def eval_snapshot(self, examinee: ExamineeBase) -> None:
        self.eval_total_score += int("2" in examinee.query_from_judge("1+1=?"))
        super().eval_snapshot(examinee)

    def tick(self) -> None:
        super().tick()

    def query_from_examinee(
        self, prompt: Union[str, Data, List[Dict]]
    ) -> Union[str, Data, List[Dict]]:
        return super().query_from_examinee(prompt)


class DummyExaminee(ExamineeBase):
    """DummyExaminee is a dummy examinee that interacts with a dummy judge in a trivial manner.
    It is only used for testing purposes. Do not use this examinee for real benchmarking.
    """

    def reset(self, model_name: str = None, **kwargs):
        super().reset(model_name=model_name, **kwargs)

    def query_from_judge(
        self, prompt: Union[str, Data, List[Dict]]
    ) -> Union[str, Data, List[Dict]]:
        return super().query_from_judge(prompt)

    def get_current_model(self) -> Model:
        return super().get_current_model()

    def run(self, judge: JudgeBase) -> Iterable:
        """Run the examinee and interact with the judge. This method is called by the user to evaluate the examinee.
        The method returns an iterable object that can be used to iterate through the examinee's interaction with the judge.
        Every iteration corresponds to the passing of a timestep."""

        # Do some initializations of self.current_model by calling base class implementation
        super().run(judge)

        while True:
            greetings: str = judge.query_from_examinee("Hello!")
            self.current_timestep += 1
            print(greetings)
            yield self.current_timestep
