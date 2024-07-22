import math
from src.abstractions.data import Data
from src.evaluation.quantify import calculate_model
import src.evaluation.utils as eval_utils
from typing import Dict, Any, Literal, Optional, List, Union, Callable
import os
import json
import re
import torch
import warnings
import src.text_writer as tw
import random
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    pipeline,
)
import numpy as np
from string import Template

from datasets import load_dataset, Dataset
from trl import (
    RewardTrainer,
    PPOConfig,
    PPOTrainer,
    RewardConfig,
    AutoModelForCausalLMWithValueHead,
)

import shutil
import gc
import subprocess
import ray

from src.abstractions.configs.templates_configs import *
import importlib
import torch.distributed as dist

import multiprocessing

# create output directories
os.makedirs("./output/benchmark_results", exist_ok=True)
os.makedirs("./output/datasets", exist_ok=True)
os.makedirs("./output/evaluation_results", exist_ok=True)
os.makedirs("./output/inference_results", exist_ok=True)
os.makedirs("./output/training_results", exist_ok=True)
os.makedirs("./output/rlhf_results", exist_ok=True)
os.makedirs("./output/merged_lora_results", exist_ok=True)


# dynamically import vllm
def import_from_vllm() -> tuple:
    vllm_lib = importlib.import_module("vllm")
    importlib.reload(vllm_lib)
    LLM, SamplingParams = vllm_lib.LLM, vllm_lib.SamplingParams

    try:
        # For vllm > 0.4.0.post1
        parallel_state_module = importlib.import_module(
            "vllm.distributed.parallel_state"
        )
    except ModuleNotFoundError:
        # For vllm == 0.4.0.post1
        parallel_state_module = importlib.import_module(
            "vllm.model_executor.parallel_utils.parallel_state"
        )

    destroy_model_parallel = getattr(parallel_state_module, "destroy_model_parallel")
    return LLM, SamplingParams, destroy_model_parallel


# escape spaces in paths
def escape(path: str):
    return path.strip().replace(" ", "\\ ")


# executes a command in terminal
def execute(command: str):
    # Print the current process ID
    print(f"Parent process ID: {os.getpid()}")
    print(f'Parent PATH: {os.environ["PATH"]}')
    subprocess.run(command, shell=True, check=True)
    print(f"End subprocess run.")


port_num = 6495


# Run the ray stop command to restart Ray cluster, in order to free up GPU memory
def restart_ray_cluster(stop_only: bool = False):
    try:
        ray.shutdown()
        subprocess.run(["ray", "stop"], check=True)
        print("Successfully stopped all Ray clusters.")
    except subprocess.CalledProcessError as e:
        print(f"Error stopping Ray clusters: {e}")

    if not stop_only:
        global port_num
        port_num -= 1
        result = subprocess.run(
            ["ray", "start", "--head", f"--port={port_num}"],
            check=True,
            capture_output=True,
            text=True,
        )
        addr = (
            [line for line in result.stdout.split("\n") if "Local node IP:" in line][0]
            .strip()
            .split(" ")[-1]
        )
        os.environ["RAY_ADDRESS"] = f"ray://{addr}:{port_num}"
        print(f"Ray address set to {os.environ['RAY_ADDRESS']}.")
        print("Successfully re-started the Ray cluster.")

    dist.init_process_group()


# vllm inference in a separate process
def vllm_inference_standalone(
    data_path: str,
    result_data_name: str,
    model_path: str,
    template_type: str,
    num_gpus: int,
    prompt_field_name: str,
    query_field_name: str,
    temperature: float,
    conn: multiprocessing.connection.Connection,
):
    LLM, SamplingParams, destroy_model_parallel = import_from_vllm()

    parallel_size = num_gpus
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        parallel_size = min(
            parallel_size, len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        )

    # truncate parallel_size to the nearest power of 2
    parallel_size = 2 ** int(math.log2(parallel_size) + 1e-7)
    print(f"vllm tensor_parallel_size = {parallel_size}")

    vllm_model = LLM(
        model=model_path,
        tensor_parallel_size=parallel_size,
        gpu_memory_utilization=0.75,
    )

    def vllm_process_batch(sample_dicts: List[dict]) -> List[dict]:
        sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=1024)
        prompts = [
            fill_in_QA_template(
                dic["instruction"], dic["input"], template_type=template_type
            )
            for dic in sample_dicts
        ]
        outputs = vllm_model.generate(prompts, sampling_params)
        assert len(outputs) == len(sample_dicts)

        for dic, output in zip(sample_dicts, outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            dic["predict"] = generated_text

        return sample_dicts

    data = Data(data_name="temporary", data_type="sft", data_path=data_path)
    data.set_key_fields(
        prompt_field_name=prompt_field_name, query_field_name=query_field_name
    )
    result_data = data.transform(
        transformation=vllm_process_batch,
        result_data_name=result_data_name,
        forced_rewrite=(
            Model.always_force_rewrite
            if hasattr(Model, "always_force_rewrite")
            else False
        ),
        max_batch_size=262144,
    )
    print("Job finished.")
    conn.send(result_data.data_path)


def fill_in_QA_template(
    instruction: str,
    input: str = "",
    suffix: str = "",
    template_type: Literal["alpaca", "mistral"] = "alpaca",
) -> str:
    """Provided with a task instruction and (optionally) supplementary input, fill them into a QA template and return the resulting prompt."""

    instruction = instruction.strip()
    input = input.strip()

    if template_type == "alpaca":
        if suffix:
            warnings.warn(f"Suffix not supported in alpaca template. Ignoring suffix.")

        input_ins = "### Instruction:\n" + instruction + "\n\n"
        if input != "":
            input_instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
            input_in = "### Input:\n" + input + "\n\n"
            input_full = input_instruction + input_ins + input_in + "### Response:\n"
        else:
            input_instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            input_full = input_instruction + input_ins + "### Response:\n"

    elif template_type == "mistral":
        input_full = f"""<s>[INST] {instruction}"""
        if input:
            input_full += f"""\n\nInput Text: \"\"\"
{input.strip()}
\"\"\""""
        if suffix:
            input_full += f"""\n\n{suffix}"""
        input_full += """ [/INST]"""

    else:
        raise NameError(
            f'Template type {template_type} not recognized. Options are "alpaca" and "mistral".'
        )

    return input_full


class Model:
    # mapping from model name to Model instance (used Any due to typing constraints), updated on the fly
    name2model: Dict[str, Any] = {}
    always_force_rewrite: bool = True

    # check with user before removing a file
    @classmethod
    def ask_and_remove_if_exists(cls, path: str, forced_rewrite: bool):
        if os.path.exists(path):
            if forced_rewrite or (
                hasattr(cls, "always_force_rewrite") and cls.always_force_rewrite
            ):
                print(f"Forced rewrite: removing {path}.")
                execute(f'rm {"-r" if os.path.isdir(path) else ""} -f {escape(path)}')
                return

            warnings.warn(
                f"{path} already exists. Use forced_rewrite=True to force rewrite."
            )
            answer = input("Do you want to force rewrite? (yes/no/always) ").lower()
            if "n" in answer:
                return
            if "a" in answer:
                cls.always_force_rewrite = True
            execute(f'rm {"-r" if os.path.isdir(path) else ""} {escape(path)}')

    def __init__(
        self,
        model_name: str,
        is_instruct_finetuned: bool = True,
        model_path: Optional[str] = None,
        num_gpus: int = None,
        template_type: str = "alpaca",
    ):
        """
        Initialize. 
        
        :param model_name: The name of the model
        :type model_name: str

        :param is_instruct_finetuned: Indicates if the model is instruction finetuned
        :type is_instruct_finetuned: bool = True

        :param model_path: The path to the model. When model_path is omitted, the model is searched for at a set of paths, including `output/{model_name}` and `output/training_results/{model_name}`.
        :type model_path: Optional[str] = None

        :param num_gpus: Number of GPUs to use for parallel finetuning/inference. Default to the total number of gpus on the machine.
        :type num_gpus: Optional[int] = None

        :param template_type: The type of template to use
        :type template_type: str = "alpaca"

        Example: Model(model_name = 'gemma-2B_sft_19century', is_instruct_finetuned = True, model_path = './output/training_results/Gemma-2B_sft_19century/')
        Example: Model(model_name = 'gemma-2B_sft_19century', is_instruct_finetuned = True)
        """
        if not num_gpus:
            num_gpus = torch.cuda.device_count()
            # num_gpus = min(num_gpus, 4)

        # if model_name in Model.name2model:
        #     warnings.warn(f'The model name {model_name} is already in use.')

        self.num_gpus = num_gpus
        self.model_name = model_name
        self.model_path = model_path
        self.is_instruct_finetuned = is_instruct_finetuned
        self.template_type = template_type

        # if model_path is not specified, look for it in the paths specified in abstractions_config.json
        if not model_path:
            for search_path in model_search_paths:
                if os.path.exists(os.path.join(search_path, model_name)):
                    print(
                        f"Found model {model_name} at {os.path.join(search_path, model_name)}"
                    )
                    self.model_path = os.path.join(search_path, model_name)
                    break

        # model_path not found
        if self.model_path is None:
            raise FileNotFoundError(f"Model {model_name} not found.")
        elif not os.path.exists(self.model_path):
            raise FileNotFoundError(f"The model path {self.model_path} doesn't exist.")

        if model_name in Model.name2model:
            Model.name2model[model_name].append(self)
        else:
            Model.name2model[model_name] = [self]

    def deep_copy(
        self,
        dest_suffix: str = None,
        dest_full_name: str = None,
        dest_subdir: Literal["training_results", "rlhf_results"] = "rlhf_results",
        source_explicit_path: str = None,
    ) -> "Model":
        """
        Returns a deep copy of the current Model instance, with either name suffix or full name of the resulting copy supplied.
        
        :param dest_suffix: The suffix for the destination
        :type dest_suffix: Optional[str] = None

        :param dest_full_name: The full name of the destination
        :type dest_full_name: Optional[str] = None

        :param dest_subdir: The subdirectory for the destination. It can be "training_results" or "rlhf_results".
        :type dest_subdir: Literal["training_results", "rlhf_results"] = "rlhf_results"

        :param source_explicit_path: The explicit path to the source
        :type source_explicit_path: Optional[str] = None
        """

        assert not (
            dest_suffix and dest_full_name
        ), "Only one of dest_suffix and dest_full_name should be provided."
        assert (
            dest_suffix or dest_full_name
        ), "Either dest_suffix or dest_full_name should be provided."

        path = self.model_path if self.model_path else source_explicit_path
        print("deep copying to", os.path.exists(path), path)
        print("checking if path exists", os.path.exists(path))
        copied_name = (
            (str(os.path.basename(path)) + "_" + str(dest_suffix))
            if dest_suffix
            else dest_full_name
        )
        copied_path = os.path.join(os.path.join("output", dest_subdir), copied_name)
        Model.ask_and_remove_if_exists(copied_path, forced_rewrite=False)
        if not os.path.exists(copied_path):
            shutil.copytree(path, copied_path)

        return Model(
            model_name=copied_name,
            is_instruct_finetuned=self.is_instruct_finetuned,
            model_path=copied_path,
            num_gpus=self.num_gpus,
            template_type=self.template_type,
        )

    def copy(self) -> "Model":
        """Returns a shallow copy of the current Model instance."""
        return Model(
            self.model_name,
            self.is_instruct_finetuned,
            self.model_path,
            self.num_gpus,
            self.template_type,
        )

    def format_dataset(self, raw_data: dict, tokenizer):
        """
        rw:
            {"chosen":[x1, x2, ...], "rejected":[y1, y2, ...]}
        map to:
            {"input_ids_chosen":[...],
             "attention_mask_chosen":[...],
             "input_ids_rejected":[...],
             "input_ids_rejected":[...]}

        """
        kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": 512,
            "return_tensors": "pt",
        }
        print("begin tokenizing rw data")
        tokens_chosen = [
            tokenizer.encode_plus(piece, **kwargs) for piece in raw_data["chosen"]
        ]
        tokens_rejected = [
            tokenizer.encode_plus(piece, **kwargs) for piece in raw_data["rejected"]
        ]

        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0],
            "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0],
            "attention_mask_rejected": tokens_rejected["attention_mask"][0],
        }

    def finetune(
        self,
        data: Data,
        stage: Literal["sft", "pretrain", "dpo", "rlhf"],
        algo: Literal["full_param", "lora"],
        result_model_name: str,
        epochs: float = None,
        batch_size_multiplier_log2: int = 0,
        grad_accu_multiplier_log2: int = -2,
        lr: float = None,
        lr_scheduler_type: str = None,
        lr_scheduler_kwargs: dict = None,
        load_best_at_end: bool = True,
        num_nodes: int = 1,
        save_checkpoints: bool = True,
        perform_eval: bool = True,
        ppo_data: Data = None,
        backend: Literal["deepspeed", "trl"] = "deepspeed",
    ) -> "Model":
        """
        Out-of-place finetuning. Doesn't update self.

        :param data: The data to be used
        :type data: Data

        :param stage: The stage of the process. It can be "sft", "pretrain", "dpo", or "rlhf".
        :type stage: Literal["sft", "pretrain", "dpo", "rlhf"]

        :param algo: The algorithm to use. It can be "full_param" or "lora".
        :type algo: Literal["full_param", "lora"]

        :param result_model_name: The name of the resulting model
        :type result_model_name: str

        :param epochs: The number of epochs
        :type epochs: Optional[float] = None

        :param batch_size_multiplier_log2: The log base 2 of the batch size multiplier
        :type batch_size_multiplier_log2: int = 0

        :param grad_accu_multiplier_log2: The log base 2 of the gradient accumulation multiplier
        :type grad_accu_multiplier_log2: int = -2

        :param lr: The learning rate
        :type lr: Optional[float] = None

        :param lr_scheduler_type: The type of learning rate scheduler
        :type lr_scheduler_type: Optional[str] = None

        :param lr_scheduler_kwargs: Additional arguments for the learning rate scheduler
        :type lr_scheduler_kwargs: Optional[dict] = None

        :param load_best_at_end: Whether to load the best model at the end
        :type load_best_at_end: bool = True

        :param num_nodes: The number of nodes
        :type num_nodes: int = 1

        :param save_checkpoints: Whether to save checkpoints
        :type save_checkpoints: bool = True

        :param perform_eval: Whether to perform evaluation
        :type perform_eval: bool = True

        :param ppo_data: The data for PPO. `ppo_data` is only used when stage is 'rlhf', and defaults to `data`.
        :type ppo_data: Optional[Data] = None

        :param backend: The backend to use. It can be "deepspeed" or "trl".
        :type backend: Literal["deepspeed", "trl"] = "deepspeed"

        :return: Returns a Model instance with name {result_model_name}, which is the result of the finetuning.
        :rtype: Model. 
        """
        if stage == "pretrain":
            assert (
                data.data_type == "pretrain"
            ), "Data type must be pretrain for pretraining."
        elif stage == "sft":
            assert data.data_type == "sft", "Data type must be sft for SFT."
        elif stage == "dpo":
            assert (
                data.data_type == "preference"
            ), "Data type must be preference for DPO."
        elif stage == "rlhf":
            assert (
                data.data_type == "preference"
            ), "Data type must be preference for RLHF."
            assert (
                isinstance(ppo_data, Data) or ppo_data is None
            ), "For RLHF, ppo_data must be an instance of Data or not provided at all."
        else:
            raise ValueError(f"Unsupported stage {stage}.")

        if lr is None:
            lr = 1.5e-5 if stage != "dpo" else 3e-7

        if lr_scheduler_type is None:
            lr_scheduler_type = "polynomial" if stage != "dpo" else "constant"

        if epochs is None:
            epochs = 4 if stage != "dpo" else 2

        if lr_scheduler_type == "polynomial" and not lr_scheduler_kwargs:
            lr_scheduler_kwargs = {"lr_end": 5e-8, "power": (11 if epochs > 1 else 1)}

        if not (save_checkpoints and perform_eval) and load_best_at_end:
            warnings.warn(
                "load_best_at_end is ignored because save_checkpoints is False."
            )

        if os.environ.get("BATCH_SIZE_MULTIPLIER_LOG2"):
            batch_size_multiplier_log2 += int(os.environ["BATCH_SIZE_MULTIPLIER_LOG2"])

        # register dataset first
        original_registration_status = data.manage_llama_factory_registration(
            operation="add"
        )

        if stage == "rlhf":
            if backend == "trl":
                warnings.warn("TRL-based RLHF has not been extensively tested yet.")
            warnings.warn("Fine-grained training parameters are ignored for RLHF.")

            ppo_data = ppo_data or data.copy()
            result = self.__rlhf(
                data,
                result_model_name,
                ppo_data,
                epochs,
                batch_size_multiplier_log2,
                grad_accu_multiplier_log2,
                num_nodes,
                use_lora=(algo == "lora"),
            ).deep_copy(dest_full_name=result_model_name)

        elif stage in ["sft", "pretrain", "dpo"]:
            if algo == "lora":
                raise NotImplementedError("LORA finetuning not implemented yet.")
            if backend == "trl":
                raise NotImplementedError("TRL finetuning not implemented yet.")

            batch_size_multiplier_log2 -= (
                2 if stage == "dpo" else 0
            )  # DPO consumes more VRAM

            # run training
            deepspeed_args = (
                f"--num_nodes={num_nodes}  --master_addr={multinode_master_addr}  --hostfile=./src/abstractions/configs/multinode/hostfile_{num_nodes}nodes"
                if num_nodes > 1
                else ""
            )  # multinode training settings
            if not os.environ.get("CUDA_VISIBLE_DEVICES"):
                deepspeed_args += f"  --num_gpus={self.num_gpus}"
            else:
                deepspeed_args += f'  --include=localhost:{os.environ["CUDA_VISIBLE_DEVICES"].strip()}'

            cmd = bash_command_template % (
                "pa38-lf" if num_nodes == 1 else "multinode-s",  # conda environment
                deepspeed_args,  # deepspeed settings
                "./src/abstractions/configs/LF_examples/full_multi_gpu/ds_z3_config.json",  # deepspeed config; this file usable for both full_param and lora
                ("pt" if stage == "pretrain" else stage),  # stage - pt, sft, dpo
                "train",  # current operation - train or predict
                "",  # do sample; ignored here
                self.model_path,  # where to find the original model
                data.data_name,  # dataset (automatically registered in llama-factory)
                self.template_type,  # template type
                ("lora" if algo == "lora" else "full"),  # type - full_param or lora
                f"./output/training_results/{escape(result_model_name)}/",  # where to save the training results (and checkpoints etc.)
                2
                ** max(
                    0, 3 + batch_size_multiplier_log2
                ),  # per_device_train_batch_size
                2
                ** max(0, 4 + batch_size_multiplier_log2),  # per_device_eval_batch_size
                2
                ** max(
                    0,
                    4
                    - max(0, 3 + batch_size_multiplier_log2)
                    + grad_accu_multiplier_log2,
                ),  # gradient_accumulation_steps
                lr_scheduler_type,  # lr_scheduler_type
                (
                    f"\n    --lr_scheduler_kwargs '{json.dumps(lr_scheduler_kwargs)}' \\"
                    if lr_scheduler_kwargs
                    else ""
                ),  # lr_scheduler_kwargs
                0.05,  # logging steps
                (0.1 if load_best_at_end else 0.2),  # save_steps
                (0.075 if stage != "dpo" else 0),  # warmup ratio
                0.1,  # eval_steps
                lr,  # learning_rate
                "steps" if save_checkpoints else "no",  # save strategy
                "steps" if perform_eval else "no",  # eval strategy
                (
                    "\n    --load_best_model_at_end \\"
                    if (load_best_at_end and save_checkpoints and perform_eval)
                    else ""
                ),  # load_best_model_at_end
                epochs,  # num_train_epochs
                "",  # omit --temperature and use default generation settings
                "",
            )  # omit --predict_with_generate because we are doing training
            print(cmd)
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                # if CUDA_VISIBLE_DEVICES is set, we need to unset it before running the command, and restore it afterwards
                # this is because deepspeed is incompatible with CUDA_VISIBLE_DEVICES, and we have already set the --include flag
                # to specify the GPUs to use (based on CUDA_VISIBLE_DEVICES)
                cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
                del os.environ["CUDA_VISIBLE_DEVICES"]
                try:
                    execute(cmd)
                finally:
                    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            else:
                execute(cmd)

            print(
                stage + " model saved at ",
                f"./output/training_results/{escape(result_model_name)}/",
            )
            result = Model(
                model_name=result_model_name,
                is_instruct_finetuned=(self.is_instruct_finetuned or stage == "sft"),
                model_path=f"./output/training_results/{escape(result_model_name)}/",
                num_gpus=self.num_gpus,
            )

        # then restore original registration status
        if not original_registration_status:
            data.manage_llama_factory_registration(operation="remove")

        # for LORA models, merge the adapter with the model
        if algo == "lora":
            print("Merging LORA model...")
            merged_model_path = f"./output/merged_lora_results/{result_model_name}"
            cmd = bash_command_for_lora_merging % (
                "pa38-lf",
                self.model_path,
                result.model_path,
                self.template_type,
                merged_model_path,
            )
            print(cmd)
            execute(cmd)
            print("Done. LORA model merged at ", merged_model_path)

            result = Model(
                model_name=result_model_name,
                is_instruct_finetuned=result.is_instruct_finetuned,
                model_path=merged_model_path,
                num_gpus=self.num_gpus,
                template_type=self.template_type,
            )

        return result

    def __rlhf(
        self,
        rw_data: Data,
        code: str,
        ppo_data: Data,
        epochs: float = 4,
        batch_size_multiplier_log2: int = 0,
        grad_accu_multiplier_log2: int = -2,
        num_nodes: int = 1,
        train_rw: bool = True,
        use_lora: bool = False,
        backend: Literal["deepspeed", "trl"] = "deepspeed",
    ):

        rw_results = "./" + os.path.join(
            "output",
            "rlhf_results",
            os.path.basename(self.model_path) + "_reward_" + code + "_results",
        )
        rw_model_copied = self.deep_copy(dest_suffix="_reward_" + code)
        rw_path, rw_name = rw_model_copied.model_path, rw_model_copied.model_name

        if backend == "deepspeed":
            deepspeed_args = (
                f"--num_nodes={num_nodes}  --master_addr={multinode_master_addr}  --hostfile=./src/abstractions/configs/multinode/hostfile_{num_nodes}nodes"
                if num_nodes > 1
                else ""
            )  # multinode training settings
            if not os.environ.get("CUDA_VISIBLE_DEVICES"):
                deepspeed_args += f"  --num_gpus={self.num_gpus}"
            else:
                deepspeed_args += f'  --include=localhost:{os.environ["CUDA_VISIBLE_DEVICES"].strip()}'

            cmd = bash_command_for_rw % (
                "pa38-lf" if num_nodes == 1 else "multinode",  # conda environment
                deepspeed_args,  # deepspeed settings
                # './src/abstractions/configs/LF_examples/deepspeed/ds_z2_config.json',
                "./src/abstractions/configs/LF_examples/full_multi_gpu/ds_z3_config.json",
                rw_path,
                rw_data.data_name,
                self.template_type,
                rw_results,
                2
                ** max(
                    0, 3 + batch_size_multiplier_log2
                ),  # per_device_train_batch_size
                2
                ** max(0, 4 + batch_size_multiplier_log2),  # per_device_eval_batch_size
                2
                ** max(
                    0,
                    4
                    - max(0, 3 + batch_size_multiplier_log2)
                    + grad_accu_multiplier_log2,
                ),  # gradient_accumulation_steps
                "polynomial",  # lr_scheduler_type
                (
                    f'\n    --lr_scheduler_kwargs \'{json.dumps({"lr_end": 1e-8, "power": 3})}\' \\'
                    if True
                    else ""
                ),  # lr_scheduler_kwargs
                epochs * 1.8,
            )
            print(cmd)
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                # if CUDA_VISIBLE_DEVICES is set, we need to unset it before running the command, and restore it afterwards
                # this is because deepspeed is incompatible with CUDA_VISIBLE_DEVICES, and we have already set the --include flag
                # to specify the GPUs to use (based on CUDA_VISIBLE_DEVICES)
                cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
                del os.environ["CUDA_VISIBLE_DEVICES"]
                try:
                    execute(cmd)
                finally:
                    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            else:
                execute(cmd)
            print("Reward modeling completed, saving to", rw_results)

        elif backend == "trl":
            if not train_rw:
                print("skipped rw_training.")
                rw_model = AutoModelForSequenceClassification.from_pretrained(rw_path)
                print("loaded")

            else:
                rw_model = AutoModelForSequenceClassification.from_pretrained("gpt2")
                rw_tokenizer = AutoTokenizer.from_pretrained("gpt2")

                if rw_tokenizer.pad_token is None:
                    rw_tokenizer.pad_token = rw_tokenizer.eos_token
                    rw_model.config.pad_token_id = rw_model.config.eos_token_id

                def formatting_func_real(examples):
                    kwargs = {
                        "padding": "max_length",
                        "truncation": True,
                        "max_length": 512,
                        "return_tensors": "pt",
                    }
                    tokens_chosen = rw_tokenizer.encode_plus(
                        examples["chosen"], **kwargs
                    )
                    tokens_rejected = rw_tokenizer.encode_plus(
                        examples["rejected"], **kwargs
                    )
                    return {
                        "input_ids_chosen": tokens_chosen["input_ids"][0],
                        "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                        "input_ids_rejected": tokens_rejected["input_ids"][0],
                        "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                    }

                rw_dataset = Dataset.from_dict(rw_data)
                rw_dataset = rw_dataset.map(formatting_func_real)
                print("rw dataset done")

                rw_config = RewardConfig(
                    max_length=1024,
                    gradient_checkpointing=False,
                    remove_unused_columns=True,
                    output_dir=rw_path,
                )
                print("rw training initiate")

                rw_trainer = RewardTrainer(
                    args=rw_config,
                    model=rw_model,
                    tokenizer=rw_tokenizer,
                    train_dataset=rw_dataset,
                )
                rw_trainer.train()

                if os.path.exists(rw_path):
                    shutil.rmtree(rw_path)

                rw_trainer.save_model(rw_path)
                print("rw training complete")

        ppo_model_copied = self.deep_copy(dest_suffix=code)
        the_path, the_name = ppo_model_copied.model_path, ppo_model_copied.model_name
        print("ppo destination", the_path, the_name)

        if backend == "deepspeed":
            original_registration_status = ppo_data.manage_llama_factory_registration(
                operation="add"
            )

            cmd = bash_command_for_ppo % (
                "pa38-lf" if num_nodes == 1 else "multinode",  # conda environment
                deepspeed_args,  # deepspeed settings
                # './src/abstractions/configs/LF_examples/full_multi_gpu/ds_z3_config.json',
                "./src/abstractions/configs/LF_examples/deepspeed/ds_z2_config.json",
                self.model_path,
                rw_results,
                "lora" if use_lora else "full",
                ppo_data.data_name,
                self.template_type,
                the_path,
                2
                ** max(
                    0, 1 + batch_size_multiplier_log2
                ),  # per_device_train_batch_size
                2
                ** max(0, 2 + batch_size_multiplier_log2),  # per_device_eval_batch_size
                2
                ** max(
                    0,
                    2
                    - max(0, 1 + batch_size_multiplier_log2)
                    + grad_accu_multiplier_log2,
                ),  # gradient_accumulation_steps
                epochs / 10,
            )
            print(cmd)
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                # if CUDA_VISIBLE_DEVICES is set, we need to unset it before running the command, and restore it afterwards
                # this is because deepspeed is incompatible with CUDA_VISIBLE_DEVICES, and we have already set the --include flag
                # to specify the GPUs to use (based on CUDA_VISIBLE_DEVICES)
                cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
                del os.environ["CUDA_VISIBLE_DEVICES"]
                try:
                    execute(cmd)
                finally:
                    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            else:
                execute(cmd)

            if not original_registration_status:
                ppo_data.manage_llama_factory_registration(operation="remove")

        elif backend == "trl":

            def get_query(sample):
                if "instruction" not in sample or "input" not in sample:
                    pattern = r"Human:(.*?)Assistant"
                    matches = re.findall(pattern, sample["chosen"], re.DOTALL)
                    query_text = matches[0].strip()
                    sample["input"] = query_text
                    sample["instruction"] = (
                        "The following is one or several paragraphs of human query. Generate the most appropriate response."
                    )

                return sample

            ppo_data = ppo_data.transform(get_query, "ppo_hh")
            ppo_data.set_key_fields(
                prompt_field_name="instruction", query_field_name="input"
            )

            # PPO setting
            config = PPOConfig(
                model_name=the_name, learning_rate=1.4e-5, batch_size=128
            )
            the_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                "./" + the_path
            )
            tokenizer = AutoTokenizer.from_pretrained(the_path)
            tokenizer.pad_token = tokenizer.eos_token

            # PPO training
            rw_model = pipeline("text-classification", model=rw_path)
            dataset = load_dataset(
                "Anthropic/hh-rlhf", data_dir="harmless-base", split="train"
            )

            def tokenize(sample):
                pattern = r"Human:(.*?)Assistant"
                matches = re.findall(pattern, sample["chosen"], re.DOTALL)
                query_text = matches[0].strip()
                sample["query"] = query_text
                sample["input_ids"] = tokenizer.encode(sample["query"])
                return sample

            dataset = dataset.map(tokenize, batched=False)
            dataset = dataset.remove_columns(["rejected", "chosen"])
            print("ppo dataset done.")

            ppo_trainer = PPOTrainer(
                model=the_model,
                config=config,
                dataset=dataset,
                tokenizer=tokenizer,
            )

            # PPO loop
            generation_kwargs = {
                "min_length": -1,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
            }
            print(dataset[0], dataset[1])

            for epoch in tqdm(range(epochs), "epoch: "):
                for batch in tqdm(ppo_trainer.dataloader):
                    query_tensors = batch["input_ids"]

                    response_tensors = ppo_trainer.generate(
                        query_tensors, **generation_kwargs
                    )
                    batch["response"] = [
                        tokenizer.decode(r.squeeze()) for r in response_tensors
                    ]

                    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
                    pipe_outputs = rw_model(texts)
                    rewards = [
                        torch.tensor(output[1]["score"]) for output in pipe_outputs
                    ]

                    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
                    ppo_trainer.log_stats(stats, batch, rewards)

            ppo_trainer.save_pretrained(the_path)
            print("RLHF complete, saving result to ", the_path)

        print("PPO done, saving to ", the_path)
        return Model(
            model_name=the_name,
            is_instruct_finetuned=self.is_instruct_finetuned,
            model_path=the_path,
            num_gpus=self.num_gpus,
        )

    def inference(
        self,
        data: Union[Data, List[Dict[str, str]]],
        result_data_name: str,
        backend: Literal["vllm", "deepspeed", "serial"] = "vllm",
        batch_size_multiplier_log2: int = 0,
        temperature=0.0,
    ) -> Union[Data, List[Dict[str, str]]]:
        """Performance inference on a dataset (currently only instruction datasets are tested, with the same format as SFT datasets),
        and 

        :param data: The data to be used. It can be either a Data object or a list of dictionaries with string keys and values. The data argument can also be a List[Dict[str,str]] where each Dict containing two fields "instruction" and (optionally) "input". In this case, a List[Dict[str, str]] will be returned.
        :type data: Union[Data, List[Dict[str, str]]]

        :param result_data_name: The name of the resulting data
        :type result_data_name: str

        :param backend: The backend to use. It can be "vllm", "deepspeed", or "serial".
        :type backend: Literal["vllm", "deepspeed", "serial"] = "vllm"

        :param batch_size_multiplier_log2: The log base 2 of the batch size multiplier
        :type batch_size_multiplier_log2: int = 0

        :param temperature: The temperature parameter
        :type temperature: float = 0.0

        :return: returns the resulting dataset (completion text saved in the `predict` field of dicts; other fields are preserved).
        :rtype: Union[Data, List[Dict[str, str]]].

        *backend*: Which backend to use for inference. Options listed below, in descreasing order of speed.

             'vllm' - Recommended. Faster than the rest by >= an order of magnitude. Parallel inference using `self.num_gpus` GPUs.

             'deepspeed' - Parallel inference using `self.num_gpus` GPUs. The only backend supporting pretrain-style inference.

             'serial' - Serial inference.
        """

        tw.write_log(
            f"Inference start, with result_data_name = {result_data_name} and backend = {backend}."
        )
        input_is_data = isinstance(data, Data)

        if os.environ.get("BATCH_SIZE_MULTIPLIER_LOG2"):
            batch_size_multiplier_log2 += int(os.environ["BATCH_SIZE_MULTIPLIER_LOG2"])

        if backend == "vllm" and os.environ.get("NO_VLLM"):
            warnings.warn("vllm is disabled. Switching to deepspeed backend.")
            backend = "deepspeed"

        if input_is_data:
            assert (
                data.data_type != "pretrain" or backend == "deepspeed"
            ), "Pretrain-style inference is only supported with backend=deepspeed."
        else:
            assert isinstance(data, list)
            if not len(data):
                return data
            assert isinstance(data[0], dict) and "instruction" in data[0]

        if backend in ["vllm", "deepspeed"]:
            if not input_is_data:
                original_data, data = data, Data(
                    data_name="temporary", data_type="sft", data_content=data
                )
                data.set_key_fields(
                    prompt_field_name="instruction", query_field_name="input"
                )

            result = (
                self.__inference_parallel_vllm_segregated(
                    data, result_data_name, temperature
                )
                if backend == "vllm"
                else self.__inference_parallel_deepspeed(
                    data, result_data_name, batch_size_multiplier_log2, temperature
                )
            )

            if not input_is_data:
                result = list(result.all_passages())

        elif backend == "serial":
            warnings.warn(
                "Setting temperature is not supported for backend=serial. It's strongly commended to switch to other backends, whether or not you have more than one GPU."
            )
            result = self.__inference_serial(data, result_data_name)

        else:
            raise NameError(
                f'Backend {backend} not recognized. Options are "vllm", "deepspeed", and "serial".'
            )

        tw.write_log(
            f"Inference finished, with result_data_name = {result_data_name} and backend = {backend}."
        )

        if isinstance(result, Data) and input_is_data:
            result.key_fields.update(data.key_fields)
            result.set_key_fields(response_field_name="predict")

        return result

    def __inference_parallel_vllm_segregated(
        self, data: Data, result_data_name: str, temperature: float
    ) -> Data:
        """vllm implementation for `inference()`, but performed in a separate process to free up GPU memory. This is the recommended implementation, due to its superior speed and robustness."""
        data_path = data.data_path
        model_path = self.model_path
        template_type = self.template_type
        num_gpus = self.num_gpus
        prompt_field_name = (
            data.key_fields["prompt"] if "prompt" in data.key_fields else "instruction"
        )
        query_field_name = (
            data.key_fields["query"] if "query" in data.key_fields else "input"
        )

        # run vllm_inference_standalone in a separate process
        multiprocessing.set_start_method("spawn", force=True)
        parent_conn, child_conn = multiprocessing.Pipe()
        p = multiprocessing.Process(
            target=vllm_inference_standalone,
            args=(
                data_path,
                result_data_name,
                model_path,
                template_type,
                num_gpus,
                prompt_field_name,
                query_field_name,
                temperature,
                child_conn,
            ),
        )
        p.start()
        p.join()
        result_data_path = parent_conn.recv()
        print("Inference results saved at ", result_data_path)

        return Data(
            data_name=result_data_name, data_type="sft", data_path=result_data_path
        )

    def __inference_parallel_vllm_integrated(
        self, data: Data, result_data_name: str, temperature: float
    ) -> Data:
        """vllm implementation for `inference()`."""
        warnings.warn(
            "This implementation is not recommended due to its potential to cause memory leaks and to disrupt Ray. Use `__inference_parallel_vllm_segregated()` instead."
        )

        restart_ray_cluster(stop_only=True)

        LLM, SamplingParams, destroy_model_parallel = import_from_vllm()

        if not hasattr(self, "vllm_model"):
            print("vllm model not created yet. creating self.vllm_model ...")

            parallel_size = self.num_gpus
            if os.environ.get("CUDA_VISIBLE_DEVICES"):
                parallel_size = min(
                    parallel_size, len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
                )

            # truncate parallel_size to the nearest power of 2
            parallel_size = 2 ** int(math.log2(parallel_size) + 1e-7)
            print(f"vllm tensor_parallel_size = {parallel_size}")

            self.vllm_model = LLM(
                model=self.model_path,
                tensor_parallel_size=parallel_size,
                gpu_memory_utilization=0.75,
            )
            print("self.vllm_model created.")

        def vllm_process_batch(sample_dicts: List[dict]) -> List[dict]:
            print(f"Processing a batch of size {len(sample_dicts)}.")
            sampling_params = SamplingParams(
                temperature=temperature, top_p=0.95, max_tokens=1024
            )
            prompts = [
                fill_in_QA_template(
                    dic["instruction"], dic["input"], template_type=self.template_type
                )
                for dic in sample_dicts
            ]
            outputs = self.vllm_model.generate(prompts, sampling_params)
            assert len(outputs) == len(sample_dicts)

            for dic, output in zip(sample_dicts, outputs):
                prompt = output.prompt
                generated_text = output.outputs[0].text
                dic["predict"] = generated_text

            # display a random element to showcase results
            showcase_id = random.randint(0, len(sample_dicts) - 1)
            tw.write_log(f"Inference sample: {sample_dicts[showcase_id]}")

            return sample_dicts

        result_data = data.transform(
            transformation=vllm_process_batch,
            result_data_name=result_data_name,
            forced_rewrite=(
                Model.always_force_rewrite
                if hasattr(Model, "always_force_rewrite")
                else False
            ),
            max_batch_size=262144,
        )

        print("Job finished. Destructing the model ...")
        self.free_gpu_memory(destroy_model_parallel)

        return result_data

    def __inference_parallel_deepspeed(
        self,
        data: Data,
        result_data_name: str,
        batch_size_multiplier_log2: int = 0,
        temperature: float = 0,
    ) -> Data:
        """Deepspeed implementation for `inference()`."""

        # register dataset first
        original_registration_status = data.manage_llama_factory_registration(
            operation="add"
        )

        result_data_path = f"./output/inference_results/{escape(result_data_name)}/"

        # run prediction
        deepspeed_args = (
            f"--num_gpus={self.num_gpus}"
            if not os.environ.get("CUDA_VISIBLE_DEVICES")
            else f'--include=localhost:{os.environ["CUDA_VISIBLE_DEVICES"].strip()}'
        )
        cmd = bash_command_template % (
            "pa38-lf",  # conda environment
            deepspeed_args,  # num_gpus; only set if CUDA_VISIBLE_DEVICES is not set
            "./src/abstractions/configs/LF_examples/full_multi_gpu/ds_z3_config.json",  # deepspeed config; this file usable for both full_param and lora
            ("sft" if data.data_type != "pretrain" else "pt"),  # stage - sft or pt
            "predict",  # current operation - train or predict
            "\n--do_sample \\",  # do sample
            self.model_path,  # where to save the resulting model
            data.data_name,  # dataset (automatically registered in llama-factory)
            self.template_type,  # template type
            "full",  # type - full_param or lora; useless here
            result_data_path,  # where to save the inference results
            2 ** max(0, 3 + batch_size_multiplier_log2),  # per_device_train_batch_size
            2 ** max(0, 4 + batch_size_multiplier_log2),  # per_device_eval_batch_size
            2,  # gradient_accumulation_steps; useless here
            "cosine",  # lr_scheduler_type; useless here
            "",  # lr_scheduler_kwargs; useless here
            0.05,  # logging steps; useless here
            10000,  # save_steps; useless here
            1,  # warmup ratio; useless here
            10000,  # eval_steps; useless here
            0,  # learning_rate; useless here
            "no",  # save_strategy; useless here
            "no",  # eval_strategy; useless here
            "",  # omit --load_best_model_at_end because we are doing inference
            1,  # num_train_epochs; useless here
            f"\n--temperature {max(temperature, 0.01)} \\",  # temperature must be positive for llama-factory
            ("\n--predict_with_generate \\" if data.data_type != "pretrain" else ""),
        )
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            # if CUDA_VISIBLE_DEVICES is set, we need to unset it before running the command, and restore it afterwards
            # this is because deepspeed is incompatible with CUDA_VISIBLE_DEVICES, and we have already set the --include flag
            # to specify the GPUs to use (based on CUDA_VISIBLE_DEVICES)
            cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            del os.environ["CUDA_VISIBLE_DEVICES"]
            try:
                execute(cmd)
            finally:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        else:
            execute(cmd)

        print("Inference results saved at ", result_data_path)

        if data.data_type == "pretrain":
            warnings.warn(
                f"`data.data_type == 'pretrain'` and therefore is doing pretrain-style inference. This hasn't been tested, and is likely buggy. We recommend you to thorough testing before using it."
            )

        # then restore original registration status
        if not original_registration_status:
            data.manage_llama_factory_registration(operation="remove")

        initial_file_path = os.path.join(
            result_data_path, "generated_predictions.jsonl"
        )
        final_file_path = os.path.join(
            result_data_path, f"{escape(result_data_name)}.json"
        )

        with tw.JsonListWriter(final_file_path) as writer:
            with open(initial_file_path, "r") as results_file:
                for i, (input_dict, result) in enumerate(
                    zip(data.all_passages(), results_file)
                ):
                    if not result.strip():
                        continue

                    result_dict = json.loads(result)

                    if "predict" not in result_dict:
                        raise KeyError(
                            f"In the {i}-th entry, field `predict` not found in inference results."
                        )

                    if (
                        "label" in result_dict
                        and "response" in data.key_fields
                        and data.key_fields["response"] in input_dict
                    ):
                        output = data.key_fields["response"]
                        if (
                            result_dict["label"]
                            and result_dict["label"] != input_dict[output]
                        ):
                            raise ValueError(
                                f"{i}-th Entry in output file and input file does not match. Potentially due to training results being out-of-order. \"{result_dict['label']}\" != \"{input_dict[output]}\""
                            )

                        del result_dict[
                            "label"
                        ]  # the two are the same, so no point in keeping both

                    writer.append({**input_dict, **result_dict})

        return Data(
            result_data_name,
            ("pretrain" if data.data_type == "pretrain" else "sft"),
            final_file_path,
        )

    def __inference_serial(
        self, input_data: Union[Data, List[Dict[str, str]]], result_data_name: str
    ) -> Union[Data, List[Dict[str, str]]]:
        """Serial implementation for `inference()`."""

        data_name = result_data_name  # self.model_name + "_inference_output"
        os.makedirs(os.path.join("output", "inference_results", "inf"), exist_ok=True)
        data_path = os.path.join(
            "output", "inference_results", "inf", data_name + ".json"
        )

        with tw.JsonListWriter(
            data_path
        ) as writer:  # memory-efficient: no need to place all answers in memory
            if isinstance(input_data, Data):
                eles = tw.read_json_memory_efficient(input_data.data_path)
            else:
                eles = input_data

            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path)

            for ele in eles:

                input_full = fill_in_QA_template(
                    ele["instruction"],
                    (ele["input"] if "input" in ele else ""),
                    template_type=self.template_type,
                )
                input_ids = tokenizer(input_full, return_tensors="pt")

                outputs = model.generate(**input_ids, max_length=2048)
                response = tokenizer.decode(outputs[0])

                ele["predict"] = response.split("##response: \n")[-1]
                # ele['answer'] = response.split('\n')[-1]

                # display the first element to showcase results
                if writer.is_first:
                    tw.write_log(
                        f"Inference sample: {ele}. Raw response: {repr(response)}."
                    )

                writer.append(ele)

        result_data = Data(data_name, "sft", data_path)
        return (
            result_data
            if isinstance(input_data, Data)
            else list(result_data.all_passages())
        )

    def evaluate(self, method: Literal["fast", "dummy"] = "fast") -> np.ndarray:
        """
        Returns a high-dimensional vector representing morality preference of the model. Choose "dummy" for fast debugging runs.
        """
        if method == "dummy" or os.environ.get("DUMMY_EVAL"):
            return np.random.rand(10)
        elif method == "fast":
            return self.__evaluate_fast()
        else:
            raise NameError(
                f'Method {method} not recognized. Options are "fast" and "dummy".'
            )

    def __evaluate_fast(self) -> np.ndarray:
        if self.template_type != "alpaca":
            raise NotImplementedError(
                "Fast evaluation is only supported for models using alpaca template."
            )

        if not os.path.exists("output"):
            os.mkdir("output")
        if not os.path.exists(os.path.join("output", "evaluation_results")):
            os.mkdir(os.path.join("output", "evaluation_results"))

        # output csv goes here
        experiment_directory = os.path.join(
            "output", "evaluation_results", self.model_name + "_single"
        )
        if not os.path.exists(experiment_directory):
            os.mkdir(experiment_directory)

        evaluation_input = eval_utils.regenerate_inputs()

        print("evaluation query begins")
        evaluation_output = self.inference(
            evaluation_input, "evaluation_output_mc_" + self.model_name, backend="vllm"
        )
        print("answers at", evaluation_output.data_path)
        with open(evaluation_output.data_path, "r") as f:
            evaluation_output_data = json.load(f)
        raw_stats = eval_utils.collect(evaluation_output_data)
        with open(
            os.path.join(experiment_directory, self.model_name + "_raw.json"), "w"
        ) as f:
            json.dump(raw_stats, f)
        print("raw results saved")
        vec = calculate_model(experiment_directory, self.model_name)
        return vec

    def __evaluate_slow_moralchoice(self) -> np.ndarray:
        """
        Returns 10 dim vector representing morality preference, saves rest of information in a json at ./output/evaluation_results/model_name.json
        The dimensions are:
        1. cause death, 2. inflict pain, 3.disable, 4. restrict freedom, 5. deprive pleasure
        6. deceive, 7. cheat, 8. break promise, 9. violate law, 10. violate duties

        This method returns an incomplete result and is slow; therefore it's abandoned.
        """
        if not os.path.exists(os.path.join("output", "evaluation_results")):
            os.mkdir(os.path.join("output", "evaluation_results"))

        directory = os.path.join(
            "libs", "moralchoice", "data", "responses", self.model_name + "_single"
        )

        if os.path.exists(directory) and (
            not os.path.exists(os.path.join(directory, "low", self.model_name + ".csv"))
            or not os.path.exists(
                os.path.join(directory, "high", self.model_name + ".csv")
            )
        ):
            print(
                "previous evaluation results incomplete - removing previous results to start fresh"
            )
            shutil.rmtree(directory)

        if not os.path.exists(directory):
            print("running evaluation")
            execute(eval_command.substitute(name=self.model_name, dir=self.model_path))

        low_vec = calculate_model(
            self.model_name + "_single",
            "low",
            os.path.join("output", "evaluation_results"),
            self.model_name,
        )
        high_vec = calculate_model(
            self.model_name + "_single",
            "high",
            os.path.join("output", "evaluation_results"),
            self.model_name,
        )
        return 1 / 3 * low_vec + 2 / 3 * high_vec

    def save_permanent(
        self, saved_name: Optional[str] = None, forced_rewrite: bool = False
    ):
        """
        Model will be saved to model_save_path from abstractions_config.json.
        Without save_permanent, the model will still be present in ./output/ and can still be directly used next time without specifying the full path.
        """
        saved_name = saved_name or self.model_name
        if saved_name != self.model_name:
            warnings.warn(
                f"Saved name {saved_name} doesn't match with model_name {self.model_name}"
            )

        path = (
            saved_name
            if "/" in saved_name
            else os.path.join(model_save_path, saved_name)
        )

        # if the path already exists, ask for approval before continuing
        Model.ask_and_remove_if_exists(path, forced_rewrite)

        # copy from model_path to path
        execute(f"cp -r {escape(self.model_path)} {escape(path)}")
        print(f"Successfully saved to {path}.")

    def free_gpu_memory(self, destroy_model_parallel: Callable[[], None] = None):
        """Remove the vllm model and free vllm cache. This should wipe out all GPU memory used by self."""
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception as e:
                print(f"destroy_model_parallel fails: {type(e)} {e}")

        try:
            del self.vllm_model.llm_engine.model_executor.driver_worker
        except Exception as e:
            print(f"del model_executor.driver_worker fails: {type(e)} {e}")

        del self.vllm_model.llm_engine
        del self.vllm_model
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.distributed.destroy_process_group()
        except:
            print("No process group to destroy.")

        restart_ray_cluster()
        gc.collect()
        print("Successfully deleted the vllm model and freed the GPU memory.")
