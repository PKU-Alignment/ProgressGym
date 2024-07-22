help_info = """
Usage:
   $ python run_benchmark.py \\
        --algorithms=ALGO1[,ALGO2[,ALGO3[,...]]] \\
        --challenges=SUBTASK1[,SUBTASK2[,SUBTASK3[,...]]] \\
        --output_filename=OUTPUT_FILENAME \\
        [--output_dir=OUTPUT_DIR] (default to ./output/benchmark_results) \\
        [--judge_model_size=JUDGE_MODEL_SIZE] (70B/8B, default to 8B) \\
        [--examinee_model_size=EXAMINEE_MODEL_SIZE] (70B/8B, default to 8B) \\
        [-h | --help] \\
        [...] (additional arguments will be supplied to the algorithms and the challenges when they are instantiated; only string values are supported)
   
Examples:
    $ python run_benchmark.py \\
          --algorithms=LifelongRLHF,LifelongDPO,OPO \\
          --challenges=Follow,Predict,Coevolve \\
          --output_filename=3x3_benchmark \\
          --judge_model_size=8B \\
          --examinee_model_size=8B
          
    $ python run_benchmark.py \\
          --algorithms=Dummy \\
          --challenges=Dummy,Coevolve \\
          --output_filename=dummy_debugging_run \\
          --judge_model_size=70B \\
          --examinee_model_size=70B
    
Note that all names are case-sensitive. Dummies are for debugging purposes only.
"""

import pdb
import traceback
import argparse
import os
import sys
import time
import json
from typing import List, Dict, Any
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()

    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-h", "--help", action="store_true")
        args_help, _ = parser.parse_known_args()
        if hasattr(args_help, "help") and args_help.help:
            print(help_info)
            sys.exit(0)

        parser.add_argument("--algorithms", type=str, required=True)
        parser.add_argument("--challenges", type=str, required=True)
        parser.add_argument("--output_filename", type=str, required=True)
        parser.add_argument(
            "--output_dir",
            type=str,
            default="./output/benchmark_results",
            required=False,
        )
        args, unknownargs = parser.parse_known_args()

        kwargs: Dict[str, str] = {}
        for s in unknownargs:
            k, v = s.split("=")
            kwargs[k.strip().strip("-")] = v.strip()

        print(
            f"Captured additional arguments: {kwargs}. They will be passed to `__init__()` and `reset()` of both the judges and the examinees, as str-typed arguments."
        )

        from benchmark.framework import JudgeBase, ExamineeBase

        algorithms: List[str] = args.algorithms.split(",")
        challenges: List[str] = args.challenges.split(",")
        output_dir: str = args.output_dir

        examinees: Dict[str, ExamineeBase] = {}
        judges: Dict[str, JudgeBase] = {}

        # Dynamically importing all algorithms
        for algorithm in algorithms:
            lib = "algorithms"
            try:
                exec(f"from {lib} import {algorithm}Examinee")
            except ImportError:
                print(
                    f"Error: Class {algorithm}Examinee not found in {lib}. Did you forget to implement it?"
                )
                sys.exit(1)

            # Instantiating the algorithm
            kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            examinee: ExamineeBase = eval(f"{algorithm}Examinee({kwargs_str})")
            examinees[algorithm] = examinee

        # Dynamically importing all challenges
        for challenge in challenges:
            lib = "benchmark"
            try:
                exec(f"from {lib} import {challenge}Judge")
            except ImportError:
                print(
                    f"Error: Class {challenge}Judge not found in {lib}. Does this challenge exist?"
                )
                sys.exit(1)

            # Instantiating the challenge
            kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            judge: JudgeBase = eval(f"{challenge}Judge({kwargs_str})")
            judges[challenge] = judge

        eval_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        path = os.path.join(output_dir, f'{args.output_filename.split(".")[0]}.json')

        # Running all algorithms on all challenges
        for algorithm in algorithms:
            for challenge in challenges:
                print(f"Running {algorithm} on {challenge}...")
                examinee = examinees[algorithm]
                judge = judges[challenge]

                examinee.reset(**kwargs)
                judge.reset(**kwargs)

                start_time = time.time()
                result: Dict[str, Any] = judge.test(examinee)
                end_time = time.time()
                result["duration_seconds"] = end_time - start_time

                if algorithm not in eval_results:
                    eval_results[algorithm] = {}
                eval_results[algorithm][challenge] = result

                with open(path, "w") as f:
                    json.dump(eval_results, f)

        with open(path, "w") as f:
            json.dump(eval_results, f)

        print(
            f"""Evaluation completed. Evaluation results saved to {path}. See item 'score' for a comprehensive score for each examinee's performance in one subtask.
            However, note that when submitting to the leaderboard, the 'score' field will be ignored, and the eventual score will be calculated from scratch."""
        )

    except:
        print(f"Exception occured. Entering debugger.")

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
