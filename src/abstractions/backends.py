# Edit flashinfer cascade.py to make it compatible with Python 3.8
from src.path import root
import os

path = os.path.join(
    os.environ["CONDA_PREFIX"], "lib/python3.8/site-packages/flashinfer/cascade.py"
)
with open(path, "r") as f:
    content = f.read()
content = content.replace("list[", "List[").replace(
    "import Optional", "import List, Optional"
)
with open(path, "w") as f:
    f.write(content)

import torch
import time
import gc
import json
import pwd
from typing import List, Tuple, Literal, Union, Dict, Callable, Optional
from nvitop import GpuProcess, Device
import multiprocessing
import subprocess
from nvitop import GpuProcess, Device
import signal
import ray
import importlib
import torch.distributed as dist
import math
import warnings
import tqdm
from transformers import AutoTokenizer
import random

# create output directories
os.makedirs(f"{root}/output/benchmark_results", exist_ok=True)
os.makedirs(f"{root}/output/datasets", exist_ok=True)
os.makedirs(f"{root}/output/evaluation_results", exist_ok=True)
os.makedirs(f"{root}/output/inference_results", exist_ok=True)
os.makedirs(f"{root}/output/training_results", exist_ok=True)
os.makedirs(f"{root}/output/rlhf_results", exist_ok=True)
os.makedirs(f"{root}/output/merged_lora_results", exist_ok=True)
os.makedirs(f"{root}/output/saved/saved_model/", exist_ok=True)
os.makedirs(f"{root}/output/saved/saved_data/", exist_ok=True)
os.makedirs(f"{root}/output/downloaded", exist_ok=True)

random.seed(time.time())
MY_USERNAME = pwd.getpwuid(os.getuid()).pw_name
PORT_NUM = 17785 + random.randint(0, 2000)


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


# Run the ray stop command to restart Ray cluster, in order to free up GPU memory
def restart_ray_cluster(stop_only: bool = False):
    try:
        ray.shutdown()
        subprocess.run(["ray", "stop"], check=True)
        print("Successfully stopped all Ray clusters.")
    except subprocess.CalledProcessError as e:
        print(f"Error stopping Ray clusters: {e}")

    if not stop_only:
        global PORT_NUM
        PORT_NUM -= 1
        result = subprocess.run(
            ["ray", "start", "--head", f"--port={PORT_NUM}"],
            check=True,
            capture_output=True,
            text=True,
        )
        addr = (
            [line for line in result.stdout.split("\n") if "Local node IP:" in line][0]
            .strip()
            .split(" ")[-1]
        )
        os.environ["RAY_ADDRESS"] = f"ray://{addr}:{PORT_NUM}"
        print(f"Ray address set to {os.environ['RAY_ADDRESS']}.")
        print("Successfully re-started the Ray cluster.")

    dist.init_process_group()


# dynamically import sglang
def import_from_sglang() -> object:
    sgl = importlib.import_module("sglang")
    importlib.reload(sgl)
    return sgl


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


def kill_all_my_gpu_processes():
    devices = Device.cuda.all()
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    for device in devices:
        processes = device.processes()
        processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
        for process in processes:
            if process.username.lower() == MY_USERNAME.lower():
                print(f"Killing process {process.pid}: {process.cmdline}")
                os.kill(process.pid, signal.SIGTERM)
                os.kill(process.pid, signal.SIGINT)
                os.kill(process.pid, signal.SIGKILL)


def get_model_size(model_repoid_or_path: str) -> float:
    model_size = (
        405
        if "405b" in model_repoid_or_path.lower()
        else (
            70
            if "70b" in model_repoid_or_path.lower()
            else (
                27
                if "27b" in model_repoid_or_path.lower()
                else (
                    13
                    if "13b" in model_repoid_or_path.lower()
                    else (
                        9
                        if "9b" in model_repoid_or_path.lower()
                        else (
                            8
                            if "8b" in model_repoid_or_path.lower()
                            else (
                                7
                                if "7b" in model_repoid_or_path.lower()
                                else (
                                    4
                                    if "3.5-mini" in model_repoid_or_path.lower()
                                    else (
                                        4
                                        if "4b" in model_repoid_or_path.lower()
                                        else (
                                            3
                                            if "3b" in model_repoid_or_path.lower()
                                            else (
                                                2
                                                if "2b" in model_repoid_or_path.lower()
                                                else (
                                                    1.7
                                                    if "1.7b"
                                                    in model_repoid_or_path.lower()
                                                    else (
                                                        1.5
                                                        if "1.5b"
                                                        in model_repoid_or_path.lower()
                                                        else (
                                                            0.5
                                                            if "0.5b"
                                                            in model_repoid_or_path.lower()
                                                            else (
                                                                0.360
                                                                if "360m"
                                                                in model_repoid_or_path.lower()
                                                                else (
                                                                    0.135
                                                                    if "135m"
                                                                    in model_repoid_or_path.lower()
                                                                    else None
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return model_size


def start_inference_backend(
    model_repoid_or_path: str,
    backend_type: Literal["sglang", "vllm"] = "sglang",
    purpose: Literal["responses", "logprobs"] = "logprobs",
    silent: bool = True,
    port: int = PORT_NUM,
    num_gpus: int = None,
    template_type: Literal["auto", "alpaca", "mistral", "llama3"] = "auto",
) -> Tuple[subprocess.Popen, Callable, Callable]:
    """Start an inference backend for a given model.
    Returns a tuple containing the backend process and the function to process a batch of samples.
    When purpose is "logprobs", the returned function will return the log probability of the prompt text itself, without generating any text. The probability will be stored in the "logprob" field of the output dictionary, with all other fields staying the same.
    When purpose is "responses", the returned function will generate a response to the prompt text. The response will be stored in the "predict" field of the output dictionary, with all other fields staying the same.

    :param model_repoid_or_path: The model repo ID or path (e.g., "meta-llama/Meta-Llama-3-8B-Instruct").
    :type model_repoid_or_path: str

    :param backend_type: The type of backend to start, defaults to "sglang"
    :type backend_type: Literal["sglang", "vllm"], optional

    :param purpose: The purpose of the backend, defaults to "logprobs"
    :type purpose: Literal["responses, "logprobs"], optional

    :param silent: Whether to run the backend silently, defaults to True
    :type silent: bool, optional

    :param port: The port number to use for the backend, defaults to PORT_NUM
    :type port: int, optional

    :param num_gpus: The number of GPUs to use for the backend, defaults to None (use all available GPUs)
    :type num_gpus: int, optional

    :param template_type: The type of template to use for the backend, defaults to "auto", which uses the appropriate template (not limited to alpaca/mistral/llama3) based on the model config file
    :type template_type: Literal["auto", "alpaca", "mistral", "llama3"], optional

    :return: A tuple containing the backend process, the function to process a batch of samples (type signature: List[dict] -> List[dict], with optional metadata arguments), and the function to destroy the backend after use.
    :rtype: Tuple[subprocess.Popen, Callable, Callable]
    """
    if eval(os.environ.get("LOUD_BACKEND", "0")):
        silent = False

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if backend_type == "vllm":

        if purpose == "logprobs":
            raise ValueError("VLLM backend does not support logprobs purpose.")

        LLM, SamplingParams, destroy_model_parallel = import_from_vllm()

        if template_type == "auto":
            template_type = model_repoid_or_path

        parallel_size = num_gpus
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            parallel_size = min(
                parallel_size, len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            )

        # truncate parallel_size to the nearest power of 2
        parallel_size = 2 ** int(math.log2(parallel_size) + 1e-7)
        print(f"vllm tensor_parallel_size = {parallel_size}")

        vllm_model = LLM(
            model=model_repoid_or_path,
            tensor_parallel_size=parallel_size,
            gpu_memory_utilization=0.75,
        )

        def vllm_process_batch(
            sample_dicts: List[dict], temperature: float = 0.2, max_tokens: int = None
        ) -> List[dict]:
            nonlocal template_type
            sampling_params = SamplingParams(
                temperature=temperature, top_p=0.95, max_tokens=max_tokens
            )

            if not os.environ.get("ALLOW_EMPTY_INSTRUCTION") or not eval(
                os.environ.get("ALLOW_EMPTY_INSTRUCTION")
            ):
                found = 0
                for dic in sample_dicts:
                    if not dic.get("instruction"):
                        if not found:
                            warnings.warn(
                                'In at least one sample, "instruction" field is missing or empty. Content from the "input" field will be moved to the "instruction" field. This behavior can be disabled by ALLOW_EMPTY_INSTRUCTION=1.'
                            )
                            found = 1
                        dic["instruction"] = dic["input"]
                        del dic["input"]

            prompts = [
                fill_in_QA_template(
                    dic.get("instruction"),
                    dic.get("input"),
                    model_repoid_or_path=template_type,
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

        def vllm_free_gpu_memory():
            """Remove the vllm model and free vllm cache. This should wipe out all GPU memory used by self."""
            if destroy_model_parallel is not None:
                try:
                    destroy_model_parallel()
                except Exception as e:
                    print(f"destroy_model_parallel fails: {type(e)} {e}")

            nonlocal vllm_model
            try:
                del vllm_model.llm_engine.model_executor.driver_worker
            except Exception as e:
                print(f"del model_executor.driver_worker fails: {type(e)} {e}")

            del vllm_model.llm_engine
            del vllm_model
            gc.collect()
            torch.cuda.empty_cache()
            try:
                torch.distributed.destroy_process_group()
            except:
                print("No process group to destroy.")

            restart_ray_cluster()
            gc.collect()
            print("Successfully deleted the vllm model and freed the GPU memory.")

        return vllm_model, vllm_process_batch, vllm_free_gpu_memory

    elif backend_type == "sglang":

        sgl = import_from_sglang()
        if template_type != "auto":
            warnings.warn(
                f"SGLang backend only supports auto template type. Ignoring template_type={template_type}. This is not an issue if you simply intend to perform inference on HistLlama models, but may be an issue if the model is neither in the HistLlama family nor in SGLang's supported models list, in which case you may use NO_SGLANG=1 to disable sglang backend."
            )

        backend_key = f"{model_repoid_or_path}-{backend_type}-{purpose}-{num_gpus}"
        connected = False

        if os.path.exists(f"{root}/output/backend_history.json"):
            with open(f"{root}/output/backend_history.json", "r") as f:
                backend_history = json.load(f)
        else:
            backend_history = {}

        print(f"Current backend history: {backend_history}", flush=True)
        print(f"Looking for prior backend with key {backend_key}...", flush=True)

        if backend_key in backend_history:
            backend_port = backend_history[backend_key]
            print(
                f"Found prior backend with key {backend_key} at port {backend_port}.",
                flush=True,
            )

            try:
                sgl.set_default_backend(sgl.RuntimeEndpoint(f"http://localhost:{port}"))
                backend_key = None
                connected = True
                backend = None
                print("Connected to backend.", flush=True)
            except:
                del backend_history[backend_key]
                print("Failed to connect to backend. Will start a new one.", flush=True)

        if not connected:
            with open(os.devnull, "w") as devnull:
                frac_static = 0.8 if purpose == "responses" else 0.7
                prefill_size = 8192 if purpose == "responses" else 1024

                model_size = get_model_size(model_repoid_or_path)
                assert model_size is not None

                if model_size <= 10 and not os.environ.get("FORCE_TP"):
                    args = [
                        "python",
                        "-m",
                        "sglang.launch_server",
                        "--port",
                        f"{port}",
                        f"--dp",
                        f"{num_gpus}",
                        "--model",
                        model_repoid_or_path,
                        "--mem-fraction-static",
                        f"{frac_static}",
                        "--chunked-prefill-size",
                        f"{prefill_size}",
                        "--trust-remote-code",
                    ]

                else:
                    min_gpus_per_instance = (
                        2 if model_size <= 30 else 4 if model_size <= 80 else 8
                    )

                    if os.environ.get("FORCE_TP"):
                        min_gpus_per_instance = int(os.environ.get("FORCE_TP"))

                    assert num_gpus % min_gpus_per_instance == 0
                    args = [
                        "python",
                        "-m",
                        "sglang.launch_server",
                        "--port",
                        f"{port}",
                        f"--tp",
                        f"{min_gpus_per_instance}",
                        f"--dp",
                        f"{num_gpus//min_gpus_per_instance}",
                        "--model",
                        model_repoid_or_path,
                        "--mem-fraction-static",
                        f"{frac_static}",
                        "--chunked-prefill-size",
                        f"{prefill_size}",
                        "--trust-remote-code",
                    ]

                # if 'int4' not in model_repoid_or_path.lower():
                #    args += ['--quantization', 'fp8']

                if "phi" in model_repoid_or_path.lower():
                    args += ["--disable-flashinfer"]

                if "smol" in model_repoid_or_path.lower():
                    args += ["--chat-template=chatml"]

                print(
                    f"Starting backend for {model_repoid_or_path} - {args}", flush=True
                )

                if silent:
                    new_env = os.environ.copy()
                    new_env["PYTHONWARNINGS"] = "ignore"
                    backend = subprocess.Popen(
                        args, stdout=devnull, stderr=devnull, env=new_env
                    )
                else:
                    backend = subprocess.Popen(args)

                print(
                    f"Registered backend with key {backend_key} at port {port}.",
                    flush=True,
                )
                backend_history[backend_key] = port
                with open(f"{root}/output/backend_history.json", "w") as f:
                    json.dump(backend_history, f)

            # Wait for backend to start
            for _ in range(40):
                time.sleep(30)
                try:
                    print(
                        f"Trying to connect to backend (at port {port})...", flush=True
                    )
                    sgl.set_default_backend(
                        sgl.RuntimeEndpoint(f"http://localhost:{port}")
                    )
                    print("Connected to backend.", flush=True)
                    break
                except:
                    print(
                        "Failed to connect to backend (this is to be expected if backend is still starting). Retrying after 30s...",
                        flush=True,
                    )
                    pass
            else:
                raise Exception("Failed to connect to backend after 20 minutes.")

        @sgl.function
        def get_response(
            s,
            conversation: List,
            temperature: float = 0.2,
            max_tokens: int = None,
            options: list = [],
        ) -> str:
            nonlocal purpose
            last_role = None

            for turn in conversation:
                if turn["role"] == "assistant":
                    s += sgl.assistant(turn["content"])
                    last_role = "assistant"

                elif turn["role"] == "user":
                    s += sgl.user(turn["content"])
                    last_role = "user"

                elif turn["role"] == "system":
                    s += sgl.system(turn["content"])

                else:
                    raise ValueError(f"Unknown role: {turn['role']}")

            if purpose == "responses" or options:
                assert last_role == "user"
                s += sgl.assistant_begin()

            if options:
                s += sgl.gen(
                    "NA",
                    max_tokens=max(len(x) for x in options) + 10,
                    choices=options,
                )

            else:
                s += sgl.gen(
                    "NA",
                    max_tokens=(max_tokens if purpose == "responses" else 0),
                    return_logprob=(purpose == "logprobs"),
                    logprob_start_len=(None if purpose == "responses" else 0),
                    temperature=temperature,
                )

        def sglang_process_batch(
            sample_dicts: List[dict], temperature: float = 0.2, max_tokens: int = None
        ) -> List[dict]:
            """Process a batch of samples using the sglang backend.
            When purpose is "logprobs", it will return the log probability of the prompt text itself, without generating any text. The probability will be stored in the "logprob" field of the output dictionary, with all other fields staying the same.
            When purpose is "responses", it will generate a response to the prompt text. The response will be stored in the "predict" field of the output dictionary, with all other fields staying the same.
            """
            nonlocal purpose

            if not os.environ.get("ALLOW_EMPTY_INSTRUCTION") or not eval(
                os.environ.get("ALLOW_EMPTY_INSTRUCTION")
            ):
                found = 0
                for dic in sample_dicts:
                    if not dic.get("instruction"):
                        if not found:
                            warnings.warn(
                                'In at least one sample, "instruction" field is missing or empty. Content from the "input" field will be moved to the "instruction" field. This behavior can be disabled by ALLOW_EMPTY_INSTRUCTION=1.'
                            )
                            found = 1
                        dic["instruction"] = dic["input"]
                        del dic["input"]

            dialogues = dict_to_dialogue_list(sample_dicts, purpose)
            options_lists = [
                (
                    dic["predict"]
                    if "predict" in dic and isinstance(dic["predict"], list)
                    else []
                )
                for dic in sample_dicts
            ]
            output = get_response.run_batch(
                [
                    {
                        "conversation": dialogue,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "options": options,
                    }
                    for dialogue, options in zip(dialogues, options_lists)
                ],
                progress_bar=True,
            )
            assert len(output) == len(sample_dicts)

            count = 0
            max_iter = int(os.environ.get("SG_ITER", 20))
            for _ in range(max_iter):
                bad_indices = [
                    k
                    for k in range(len(output))
                    if output[k].get_meta_info("NA") is None
                ]
                if len(bad_indices) == 0:
                    break

                new_output = get_response.run_batch(
                    [
                        {
                            "conversation": dialogues[k],
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "options": options_lists[k],
                        }
                        for k in bad_indices
                    ],
                    progress_bar=True,
                )
                assert len(new_output) == len(bad_indices)

                count = 0
                for k, out in zip(bad_indices, new_output):
                    output[k] = out
                    if out.get_meta_info("NA") is None:
                        count += 1

                print(
                    f"Re-run {len(bad_indices)} cases, {count} cases still not completed."
                )
                if count == len(bad_indices):
                    warnings.warn(
                        f"Rerunning did not help. {count} cases still not completed. Use NO_SGLANG=1 to disable sglang backend."
                    )
                    break
            else:
                warnings.warn(
                    f"{count} cases still not completed after {max_iter} retries. Use NO_SGLANG=1 to disable sglang backend."
                )

            if os.environ.get("MAX_SG_FAIL", "").lower() != "inf" and count > eval(os.environ.get("MAX_SG_FAIL", "min(100, len(output)//100)")):
                raise Exception(f"Too many cases ({count}) still not completed. Aborting. Use MAX_SG_FAIL=inf to disable this check.")

            failure_count = 0
            for dic, out in zip(sample_dicts, output):
                if out.get_meta_info("NA") is None:
                    failure_count += 1
                    continue

                if purpose == "logprobs":
                    if "predict" in dic and isinstance(dic["predict"], list):
                        dic["logprob"] = [
                            sum(x[0] for x in y if x[0] is not None)
                            for y in list(
                                out.get_meta_info("NA")["input_token_logprobs"]
                            )
                        ]
                        assert len(dic["logprob"]) == len(dic["predict"])
                    else:
                        dic["logprob"] = sum(
                            x[0]
                            for x in list(
                                out.get_meta_info("NA")["input_token_logprobs"]
                            )
                            if x[0] is not None
                        )
                else:
                    dic["predict"] = (
                        out["NA"] if out.get_meta_info("NA") is not None else None
                    )

            if failure_count > count:
                raise Exception(
                    f"More actual failures ({failure_count}) than cases not completed ({count}), which is unexpected."
                )

            return sample_dicts

        def sglang_free_gpu_memory():
            """Wipe out all GPU memory used by the user."""
            nonlocal backend_key

            # Remove the backend from the history
            with open(f"{root}/output/backend_history.json", "r") as f:
                backend_history = json.load(f)

            if backend_key:
                backend_history.pop(backend_key)
            
            with open(f"{root}/output/backend_history.json", "w") as f:
                json.dump(backend_history, f)

            # Kill the backend process
            try:
                backend.kill()
            except:
                print("backend.kill() failed.")

            MY_USERNAME = pwd.getpwuid(os.getuid()).pw_name
            print(f"Killing all processes on GPU for user {MY_USERNAME}.")

            devices = Device.cuda.all()
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)
            for device in devices:
                processes = device.processes()
                processes = GpuProcess.take_snapshots(processes.values(), failsafe=True)
                for process in processes:
                    if process.username.lower() == MY_USERNAME.lower():
                        print(f"Killing process {process.pid}: {process.cmdline}")
                        os.kill(process.pid, signal.SIGTERM)
                        os.kill(process.pid, signal.SIGINT)
                        os.kill(process.pid, signal.SIGKILL)

        return backend, sglang_process_batch, sglang_free_gpu_memory

    raise ValueError(f"Backend type {backend_type} not recognized.")


def dict_to_dialogue_list(
    dic: Union[dict, List[dict]],
    purpose: Literal["responses", "logprobs"] = "responses",
) -> Union[List[Dict[str, str]], List[List[Dict[str, str]]]]:
    """Transform a dictionary into a list of dialogue turns in OpenAI format.

    :param dic: Dictionary containing 'instruction' and 'input' keys.
    :type dic: Union[dict, List[dict]]
    :return: List of dialogue turns in OpenAI format.
    :rtype: Union[List[Dict[str, str]], List[List[Dict[str, str]]]
    """
    if isinstance(dic, dict):
        res = []

        if "system" in dic:
            res = [{"role": "system", "content": dic["system"]}]

        if "history" in dic:
            for turn in dic["history"]:
                res.append({"role": "user", "content": turn[0]})
                res.append({"role": "assistant", "content": turn[1]})

        if "input" in dic or "instruction" in dic:
            input = dic.get("input", "")
            instruction = dic.get("instruction", "")
            res.append(
                {
                    "role": "user",
                    "content": input
                    + ("\n\n" if input and instruction else "")
                    + instruction,
                }
            )

        if (
            purpose == "logprobs"
            and "predict" in dic
            and isinstance(dic["predict"], str)
        ):
            res.append({"role": "assistant", "content": dic["predict"]})
        elif "output" in dic:
            res.append({"role": "assistant", "content": dic["output"]})

        return res

    return [dict_to_dialogue_list(d) for d in dic]


def fill_in_QA_template(
    instruction: str = "",
    input: str = "",
    suffix: str = "",
    full_dict: dict = None,
    model_repoid_or_path: Union[Literal["alpaca", "mistral", "llama3"], str] = "alpaca",
) -> str:
    """Provided with a task instruction and (optionally) supplementary input, fill them into a QA template and return the resulting prompt.

    :param instruction: The task instruction, defaults to "". Either this or full_dict must be provided.
    :type instruction: str, optional

    :param input: Supplementary input to the task, defaults to "".
    :type input: str, optional

    :param suffix: Suffix to add to the prompt, defaults to "".
    :type suffix: str, optional

    :param full_dict: The full dictionary containing the instruction and input, defaults to None. Either this or instruction must be provided. If this is provided, instruction, input, and suffix will be ignored.
    :type full_dict: dict, optional

    :param model_repoid_or_path: The model repo ID or path (e.g., "meta-llama/Meta-Llama-3-8B-Instruct"), or one of the special values "alpaca" or "mistral" or "llama3", defaults to "alpaca".
    :type model_repoid_or_path: Union[Literal["alpaca", "mistral", "llama3"], str], optional

    :return: The prompt with the instruction and input filled in.
    :rtype: str
    """

    instruction = instruction.strip()
    input = input.strip()

    # Convert full_dict to instruction and input
    if full_dict and model_repoid_or_path in ["alpaca", "mistral"]:
        assert (
            "history" not in full_dict
        ), "History field not supported with alpaca/mistral template."
        assert (
            "system" not in full_dict
        ), "System field not supported with alpaca/mistral template."

        instruction = full_dict.get("instruction", "")
        input = full_dict.get("input", "")

    if input and not instruction:
        warnings.warn("Swapping instruction and input fields.")
        instruction, input = input, instruction

    if model_repoid_or_path == "alpaca":
        if suffix:
            warnings.warn(
                f"Suffix not supported except with mistral template. Ignoring suffix."
            )

        input_ins = "### Instruction:\n" + instruction + "\n\n"
        if input != "":
            input_instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
            input_in = "### Input:\n" + input + "\n\n"
            input_full = input_instruction + input_ins + input_in + "### Response:\n"
        else:
            input_instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            input_full = input_instruction + input_ins + "### Response:\n"

    elif model_repoid_or_path == "mistral":
        input_full = f"""<s>[INST] {instruction}"""
        if input:
            input_full += f"""\n\nInput Text: \"\"\"
{input.strip()}
\"\"\""""
        if suffix:
            input_full += f"""\n\n{suffix}"""
        input_full += """ [/INST]"""

    else:
        if model_repoid_or_path == "llama3":
            model_repoid_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"

        if suffix:
            warnings.warn(
                f"Suffix not supported except with mistral template. Ignoring suffix."
            )

        prompt = dict_to_dialogue_list(
            full_dict if full_dict else {"instruction": instruction, "input": input}
        )
        tokenizer = AutoTokenizer.from_pretrained(model_repoid_or_path)
        input_full = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(input_full, str)

    return input_full
