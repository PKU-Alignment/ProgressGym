from abc import abstractmethod, ABC  # Abstract Base Classes
from typing import Iterable, Tuple, Dict, Literal, Any, List, Union
from src.abstractions import Model, Data, fill_in_QA_template
from time import strftime, localtime
from random import randint
import os
import traceback


class JudgeBase(ABC):
    """JudgeBase is the base class for all judges.
    A judge is the benchmarking algorithm that evaluates the performance of an examinee.
    Each judge class corresponds to a challenge."""

    """Evaluation results"""
    examinee_model_history: List[Model]
    judge_model_history: List[Model]
    eval_times: int

    """Query statistics"""
    query_times: int
    query_total_length: int

    """Current model"""
    current_model: Model
    current_timestep: int
    model_size: int

    """Information determined at initialization"""
    instance_id: str
    model_list: List[Model]
    template_type: Literal["alpaca", "mistral"]
    debug: bool
    checkpoint_id: str

    def __init__(self, **kwargs):
        self.reset(**kwargs)

    @abstractmethod
    def reset(self, **kwargs) -> None:
        """Reset the internal state of the judge to start a new evaluation. This method is called before each test.

        The base class implementation resets the internal state of the judge to the initial state.
        Normally, you should optionally call the base class implementation in your subclass's implementation, and then add
        any additional reset logic that you need."""

        """Find the list of all models, sorted by timestep."""
        self.template_type = (
            "alpaca" if "template_type" not in kwargs else kwargs["template_type"]
        )
        self.model_size = (
            int(kwargs["judge_model_size"].lower().replace("b", "").strip())
            if "judge_model_size" in kwargs
            else 8
        )
        self.model_list = []
        for i in range(22):
            try:
                model_name = "%db-C%03d-instruct" % (self.model_size, i)
                self.model_list.append(
                    Model(
                        model_name=model_name,
                        is_instruct_finetuned=True,
                        template_type=self.template_type,
                    )
                )
            except:
                pass

        """Initialize the internal language model."""
        self.current_timestep = 0
        self.current_model = self.model_list[self.current_timestep]

        """Reset the query and eval statistics."""
        self.query_total_length = 0
        self.query_times = 0
        self.eval_times = 0
        self.examinee_model_history = []
        self.judge_model_history = []

        """Set the judge instance ID."""
        self.instance_id = (
            self.__class__.__name__
            + "_"
            + strftime("%d%b%H%M%S", localtime())
            + "_"
            + str(randint(0, 10**8))
        )
        print(f"Found {len(self.model_list)} models for {self.instance_id}.")
        self.checkpoint_id = None
        if "load_checkpoint_from" in kwargs:
            for checkpoint in kwargs["load_checkpoint_from"].strip().split(","):
                if self.__class__.__name__ in checkpoint:
                    # assert os.path.exists(checkpoint), f'Checkpoint {checkpoint} does not exist.'
                    assert (
                        self.checkpoint_id is None
                    ), f"Checkpoint {self.checkpoint_id} is already loaded, but {checkpoint} is also specified."
                    self.checkpoint_id = checkpoint
                    print(f"Loaded checkpoint {checkpoint} for {self.instance_id}.")

            if self.checkpoint_id is None:
                print(f"No checkpoint found for {self.instance_id}.")

        """Set debug mode."""
        self.debug = "debug" in kwargs and eval(kwargs["debug"])

    @abstractmethod
    def eval_snapshot(self, examinee: "ExamineeBase") -> None:
        """Evaluate the examinee's performance at the current snapshot. This method is called by the judge at every iteration.

        The base class implementation only does logging. It is recommended to does your own eval and then call the base class
        implementation to perform logging."""

        self.eval_times += 1
        self.examinee_model_history.append(examinee.get_current_model())
        self.judge_model_history.append(self.current_model)

    @abstractmethod
    def tick(self) -> None:
        """Move the internal state of the judge to the next timestep. This method is called by the judge at every iteration.

        The base class implementation moves the judge to the next timestep by incrementing `current_timestep` by 1 (or more if necessary).
        You should optionally call the base class implementation in your subclass's implementation, and then add any additional
        logic that you need."""

        self.current_timestep += 1
        if self.current_timestep >= len(self.model_list):
            raise IndexError(
                f"{self.current_timestep}-th timestep doesn't exist. The model time sequence have been exhausted."
            )

        self.current_model = self.model_list[self.current_timestep]

    @abstractmethod
    def query_from_examinee(
        self, prompt: Union[str, Data, List[Dict]], model: Model = None
    ) -> Union[str, Data, List[Dict]]:
        """This method is called by the examinee to query the judge, which the judge will answer according to human preferences at the current timestep.
        The examinee will use this information to learn about the latest human preference, and update its language model accordingly.

        The base class implementation answers the prompt by directly querying `self.current_model``
        You could either call the base class implementation in your subclass's implementation (possibly supplying a different `model`),
        or override it if necessary."""

        model = model or self.current_model

        if isinstance(prompt, str):
            self.query_times += 1
            self.query_total_length += len(prompt)
            try:
                result_data = Data(
                    f"{self.checkpoint_id}_query_from_examinee_{self.query_times}th"
                )
                results: str = list(result_data.all_passages())[0]["predict"]
                print(f"Found query result from {result_data.data_path}.")
            except:
                results: str = model.inference(
                    data=[{"instruction": prompt}],
                    result_data_name=f"{self.instance_id}_query_from_examinee_{self.query_times}th",
                    backend="vllm",
                )[0]["predict"]

        elif isinstance(prompt, Data):
            cnt = len(list(prompt.all_passages()))
            self.query_times += cnt
            self.query_total_length += (
                len(next(prompt.all_passages())) * cnt
            )  # only an estimate
            try:
                result_data = Data(
                    f"{self.checkpoint_id}_query_from_examinee_{self.query_times}th"
                )
                results: Data = result_data
                results.key_fields.update(prompt.key_fields)
                results.set_key_fields(response_field_name="predict")
                print(f"Found query result from {result_data.data_path}.")
            except:
                results: Data = model.inference(
                    data=prompt,
                    result_data_name=f"{self.instance_id}_query_from_examinee_{self.query_times}th",
                    backend="vllm",
                )

        elif isinstance(prompt, list):
            cnt = len(prompt)
            self.query_times += cnt
            self.query_total_length += sum(
                len(dic["instruction"]) + len(dic["input"]) for dic in prompt
            )
            try:
                result_data = Data(
                    f"{self.checkpoint_id}_query_from_examinee_{self.query_times}th"
                )
                results: List[Dict] = list(result_data.all_passages())
                print(f"Found query result from {result_data.data_path}.")
            except:
                results: List[Dict] = model.inference(
                    data=prompt,
                    result_data_name=f"{self.instance_id}_query_from_examinee_{self.query_times}th",
                    backend="vllm",
                )

        else:
            raise ValueError(
                "prompt must be either a string, a Data object, or a list of dictionaries."
            )

        return results

    @abstractmethod
    def produce_final_result(self) -> Dict[str, Any]:
        """Return the final result of the evaluation. This method is called at the end of `test()` to get the final evaluation metrics.
        A reference score may be calculated here, but it will not be used by the leaderboard, in order to prevent manual score manipulation.

        The base class implementation only performs logging. You should override this method in your subclass to fill in the evaluation metrics, while preserving logging-purposed dict fields returned by the base class implementation.
        """

        return {
            "eval_times": self.eval_times,
            "query_times": self.query_times,
            "query_total_length": self.query_total_length,
            "examinee_model_at_each_timestep": [
                model.model_path for model in self.examinee_model_history
            ],
            "judge_model_at_each_timestep": [
                model.model_path for model in self.judge_model_history
            ],
        }

    @abstractmethod
    #@classmethod
    def interpret_result(cls, result: Dict[str, Any]) -> float:
        """Given an benchmark result dictionary, calculate a single score that represents the overall performance of the examinee. HIGHER scores must mean better performance. This method is called by the leaderboard to rank the examinees."""
        raise NotImplementedError

    def test(self, examinee: "ExamineeBase", **kwargs) -> Dict[str, Any]:
        """Run the examinee and evaluate its performance. This method is called by the user to evaluate the examinee.
        The method returns a dictionary of evaluation metrics. The keys of the dictionary are the names of the metrics, and the values are the corresponding values of the metrics.
        The method operates by moving the examinee and the judge through a series of timesteps, where the judge evaluates the examinee at every timestep.
        Every iteration of examinee_iter corresponds to the passing of a timestep.

        Normally, you should not override this method in your subclass. Instead, you should implement the `reset`, `eval_snapshot`, `tick`, `query_from_examinee`, and `produce_final_result` methods in your subclass.
        """

        def test_loop() -> Dict[str, Any]:
            examinee_iter = examinee.run(self)
            self.tick()

            for _ in examinee_iter:
                print(f"Judge at {self.current_timestep}th timestep.")
                self.eval_snapshot(examinee)
                try:
                    self.tick()
                except:
                    break

            return self.produce_final_result()

        if self.debug:
            # Do not catch exceptions, to allow for interactive debugging
            return test_loop()

        try:
            return test_loop()
        except Exception as e:
            print(f"Halting test due to error: {type(e)} {e}")
            traceback.print_exc()
            trace_str = traceback.format_exc()
            return {"error": f"{type(e)} {e} {trace_str}"}


class ExamineeBase(ABC):
    """ExamineeBase is the base class for all examinees.
    An examinee is the an alignment algorithm (in combination with a language model operated upon by the algorithm) that is benchmarked by a judge.
    You are free to implement the benchmarked examinee in any way you like, as long as it follows the ExamineeBase interface.
    In most cases, you need to re-implement most or all all the methods in your subclass. Base implementations are only provided as an example.
    """

    """Current model"""
    current_model: Model
    current_timestep: int
    template_type: Literal["alpaca", "mistral"]

    """Information determined at initialization"""
    instance_id: str
    checkpoint_id: str

    """Query statistics"""
    query_times: int

    def __init__(self, **kwargs):
        self.reset(**kwargs)

    @abstractmethod
    def reset(self, **kwargs) -> None:
        """Initialize the examinee, including endowing it with a language model.

        When `examinee_model_size` is not specified, the model will be initialized as a copy of the Judge's initial model. In that case, the examinee will be able to start from the same initial state as the judge.
        Normally, you should implement this method in your subclass to initialize the examinee as needed, after calling the base class implementation for basic setup.
        """
        if "model_name" not in kwargs:
            self.model_size = (
                int(kwargs["examinee_model_size"].lower().replace("b", "").strip())
                if "examinee_model_size" in kwargs
                else 8
            )
            for i in range(22):
                try:
                    kwargs["model_name"] = "%db-C%03d-instruct" % (self.model_size, i)
                except:
                    pass

        self.current_model = (
            Model(kwargs["model_name"])
            if "model_name" in kwargs and kwargs["model_name"]
            else None
        )
        self.current_timestep = 0
        self.template_type = "alpaca"
        self.instance_id = (
            self.__class__.__name__
            + "_"
            + strftime("%d%b%H%M%S", localtime())
            + "_"
            + str(randint(0, 10**8))
        )
        self.query_times = 0

        self.checkpoint_id = None
        if "load_checkpoint_from" in kwargs:
            for checkpoint in kwargs["load_checkpoint_from"].strip().split(","):
                if self.__class__.__name__ in checkpoint:
                    # assert os.path.exists(checkpoint), f'Checkpoint {checkpoint} does not exist.'
                    assert (
                        self.checkpoint_id is None
                    ), f"Checkpoint {self.checkpoint_id} is already loaded, but {checkpoint} is also specified."
                    self.checkpoint_id = checkpoint
                    print(f"Loaded checkpoint {checkpoint} for {self.instance_id}.")

            if self.checkpoint_id is None:
                print(f"No checkpoint found for {self.instance_id}.")

    @abstractmethod
    def query_from_judge(
        self, prompt: Union[str, Data, List[Dict]], model: Model = None
    ) -> Union[str, Data, List[Dict]]:
        """This method is called by the judge to query the examinee for a response to a prompt.

        In most cases, you only need to call the base class implementation in your subclass's implementation.
        """

        model = model or self.current_model
        self.query_times += 1

        try:
            result_data = Data(
                f"{self.checkpoint_id}_query_from_judge_{self.query_times}th"
            )

            if isinstance(prompt, str):
                results: str = list(result_data.all_passages())[0]["predict"]

            elif isinstance(prompt, Data):
                results: Data = result_data
                results.key_fields.update(prompt.key_fields)
                results.set_key_fields(response_field_name="predict")

            elif isinstance(prompt, list):
                results: List[Dict] = list(result_data.all_passages())

            else:
                raise ValueError(
                    "prompt must be either a string, a Data object, or a list of dictionaries."
                )

            print(f"Found query result from {result_data.data_path}.")

        except:

            if isinstance(prompt, str):
                results: str = model.inference(
                    data=[{"instruction": prompt, "input": ""}],
                    result_data_name=f"{self.instance_id}_query_from_judge_{self.query_times}th",
                    backend="vllm",
                )[0]["predict"]

            elif isinstance(prompt, Data):
                results: Data = model.inference(
                    data=prompt,
                    result_data_name=f"{self.instance_id}_query_from_judge_{self.query_times}th",
                    backend="vllm",
                )

            elif isinstance(prompt, list):
                results: List[Dict] = model.inference(
                    data=prompt,
                    result_data_name=f"{self.instance_id}_query_from_judge_{self.query_times}th",
                    backend="vllm",
                )

            else:
                raise ValueError(
                    "prompt must be either a string, a Data object, or a list of dictionaries."
                )

        return results

    @abstractmethod
    def get_current_model(self) -> Model:
        """Return the current model that the examinee is using at this timestep.

        The base class implementation returns the `current_model` attribute.
        You should not need to override this method in your subclass unless the model is not stored in the `current_model` attribute.
        """

        return self.current_model

    @abstractmethod
    def run(self, judge: JudgeBase) -> Iterable:
        """This method is called by the judge to start the examinee.
        It will return an iterable that the judge will iterate over to run the examinee.
        Every iteration corresponds to the passing of a timestep.
        In this way, the examinee can control the pause and resume of the examinee.
        At every iteration:
          1. The examinee learns about the latest human preference by calling the judge's query_from_examinee method.
          2. After it has updated its language model, it yields control back to the judge and allow it to evaluate it (by calling query_from_judge).
        Unless you are sure that you need to completely override this method, you should not do so. Instead, call the base class implementation at the beginning of your subclass's implementation.
        """

        # Initialize the examinee's model to match it with the starting point of the Judge
        if not self.current_model:
            self.current_model = judge.model_list[self.current_timestep].copy()

        # Example of what should come afterwards. This should be implemented in the subclass implementation.
        #
        # while True:
        #     self.current_timestep += 1
        #     (TODO: obtain human preference by calling judge.query_from_examinee, then update self.current_model accordingly)
        #     yield self.current_timestep
