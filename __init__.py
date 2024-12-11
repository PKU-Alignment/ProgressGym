import os, sys
sys.path = [os.path.dirname(os.path.abspath(__file__))] + sys.path

if not eval(os.environ.get("LOUD_BACKEND", "0")):
    os.environ["WANDB_DISABLED"] = "true"

import logging
logging.basicConfig(level=logging.ERROR)

from benchmark.framework import JudgeBase, ExamineeBase
from benchmark.dummies import DummyJudge
from challenges.follow import FollowJudge
from challenges.predict import PredictJudge
from challenges.coevolve import CoevolveJudge

from algorithms.lifelong_dpo import LifelongDPOExaminee
from algorithms.lifelong_rlhf import LifelongRLHFExaminee
from algorithms.extrapolative_dpo import ExtrapolativeDPOExaminee
from algorithms.extrapolative_rlhf import ExtrapolativeRLHFExaminee
from benchmark.dummies import DummyExaminee

from run_benchmark import run_benchmark

from src.abstractions.model import Model, fill_in_QA_template
from src.abstractions.data import Data, DataFileCollection
from src.abstractions.configs.templates_configs import GlobalState

__all__ = [
    "run_benchmark",
    "Model",
    "Data",
    "DataFileCollection",
    "JudgeBase",
    "ExamineeBase",
    "DummyJudge",
    "FollowJudge",
    "PredictJudge",
    "CoevolveJudge",
    "DummyExaminee",
    "LifelongDPOExaminee",
    "LifelongRLHFExaminee",
    "ExtrapolativeDPOExaminee",
    "ExtrapolativeRLHFExaminee",
    "fill_in_QA_template",
    "GlobalState",
]
