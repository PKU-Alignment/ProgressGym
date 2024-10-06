# Edit flashinfer cascade.py to make it compatible with Python 3.8
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
import pwd
from typing import List, Tuple, Literal, Union, Dict, Callable, Optional
import multiprocessing
import subprocess
from nvitop import GpuProcess, Device
import signal
import importlib
import ray
import importlib
import torch.distributed as dist
import math
import warnings
import tqdm
from transformers import AutoTokenizer

# create output directories
os.makedirs("./output/benchmark_results", exist_ok=True)
os.makedirs("./output/datasets", exist_ok=True)
os.makedirs("./output/evaluation_results", exist_ok=True)
os.makedirs("./output/inference_results", exist_ok=True)
os.makedirs("./output/training_results", exist_ok=True)
os.makedirs("./output/rlhf_results", exist_ok=True)
os.makedirs("./output/merged_lora_results", exist_ok=True)
os.makedirs("./output/saved/saved_model/", exist_ok=True)
os.makedirs("./output/saved/saved_data/", exist_ok=True)
os.makedirs("./output/downloaded", exist_ok=True)

MY_USERNAME = pwd.getpwuid(os.getuid()).pw_name
PORT_NUM = 14285


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
    template_type: Literal["auto", "alpaca", "mistral"] = "auto",
) -> Tuple[subprocess.Popen, Callable]:
    """Start an inference backend for a given model.

    :param model_repoid_or_path: The model repo ID or path (e.g., "meta-llama/Llama-3.1-8B-Instruct").
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
    :param template_type: The type of template to use for the backend, defaults to "auto", which uses the appropriate template (not limited to alpaca/mistral) based on the model config file
    :type template_type: Literal["auto", "alpaca", "mistral"], optional
    :return: A tuple containing the backend process and the function to process a batch of samples (type signature: List[dict] -> List[dict], with optional metadata arguments)
    :rtype: Tuple[subprocess.Popen, Callable]
    """

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if backend_type == "vllm":

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
            sample_dicts: List[dict], temperature: float = 0.2, max_tokens: int = 1024
        ) -> List[dict]:
            nonlocal template_type
            sampling_params = SamplingParams(
                temperature=temperature, top_p=0.95, max_tokens=max_tokens
            )
            
            if not os.environ.get("ALLOW_EMPTY_INPUT") or not eval(os.environ.get("ALLOW_EMPTY_INPUT")):
                found = 0
                for dic in sample_dicts:
                    if not dic.get("input"):
                        if not found:
                            warnings.warn('In at least one sample, "input" field is missing or empty. Content from the "instruction" field will be copied to the "input" field. This behavior can be disabled by ALLOW_EMPTY_INPUT=1.')
                            found = 1
                        dic["input"] = dic["instruction"]

            prompts = [
                fill_in_QA_template(
                    dic["instruction"], dic["input"], model_repoid_or_path=template_type
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

        return vllm_model, vllm_process_batch

    elif backend_type == "sglang":

        sgl = import_from_sglang()
        if template_type != "auto":
            warnings.warn(
                f"SGLang backend only supports auto template type. Ignoring template_type={template_type}. This is not an issue if you simply intend to perform inference on HistLlama models, but may be an issue if you are using a custom model, in which case you may use NO_SGLANG=1 to disable sglang backend."
            )

        with open(os.devnull, "w") as devnull:
            frac_static = 0.8 if purpose == "responses" else 0.4
            prefill_size = 8192 if purpose == "responses" else 1024

            model_size = get_model_size(model_repoid_or_path)
            assert model_size is not None

            if model_size <= 10:
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

            print(f"Starting backend for {model_repoid_or_path} - {args}", flush=True)

            if silent:
                backend = subprocess.Popen(args, stdout=devnull, stderr=devnull)
            else:
                backend = subprocess.Popen(args)

        # Wait for backend to start
        for _ in range(40):
            time.sleep(30)
            try:
                print("Trying to connect to backend...", flush=True)
                sgl.set_default_backend(sgl.RuntimeEndpoint(f"http://localhost:{port}"))
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
            s, conversation: List, temperature: float = 0.2, max_tokens: int = 256
        ) -> str:

            for turn in conversation:
                if turn["role"] == "assistant":
                    s += sgl.assistant(turn["content"])
                elif turn["role"] == "user":
                    s += sgl.user(turn["content"])
                elif turn["role"] == "system":
                    s += sgl.system(turn["content"])
                else:
                    raise ValueError(f"Unknown role: {turn['role']}")

            s += sgl.assistant_begin()
            s += sgl.gen(
                "NA",
                max_tokens=max_tokens,
                return_logprob=False,
                temperature=temperature,
            )

        def sglang_process_batch(
            sample_dicts: List[dict], temperature: float = 0.2, max_tokens: int = 256
        ) -> List[dict]:
            if not os.environ.get("ALLOW_EMPTY_INPUT") or not eval(os.environ.get("ALLOW_EMPTY_INPUT")):
                found = 0
                for dic in sample_dicts:
                    if not dic.get("input"):
                        if not found:
                            warnings.warn('In at least one sample, "input" field is missing or empty. Content from the "instruction" field will be copied to the "input" field. This behavior can be disabled by ALLOW_EMPTY_INPUT=1.')
                            found = 1
                        dic["input"] = dic["instruction"]
            
            dialogues = dict_to_dialogue_list(sample_dicts)
            output = get_response.run_batch(
                [
                    {
                        "conversation": dialogue,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    for dialogue in dialogues
                ],
                progress_bar=True,
            )
            assert len(output) == len(sample_dicts)

            for _ in range(20):
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
                    f"{count} cases still not completed after 10 retries. Use NO_SGLANG=1 to disable sglang backend."
                )

            for dic, out in zip(sample_dicts, output):
                dic["predict"] = (
                    out["NA"] if out.get_meta_info("NA") is not None else None
                )

            return sample_dicts

        return backend, sglang_process_batch

    raise ValueError(f"Backend type {backend_type} not recognized.")


def dict_to_dialogue_list(
    dic: Union[dict, List[dict]]
) -> Union[List[Dict[str, str]], List[List[Dict[str, str]]]]:
    """Transform a dictionary into a list of dialogue turns in OpenAI format.

    :param dic: Dictionary containing 'instruction' and 'input' keys.
    :type dic: Union[dict, List[dict]]
    :return: List of dialogue turns in OpenAI format.
    :rtype: Union[List[Dict[str, str]], List[List[Dict[str, str]]]
    """
    if isinstance(dic, dict):
        return [
            {"role": "system", "content": dic["instruction"]},
            {"role": "user", "content": dic["input"]},
        ]

    return [dict_to_dialogue_list(d) for d in dic]


def fill_in_QA_template(
    instruction: str,
    input: str = "",
    suffix: str = "",
    model_repoid_or_path: Union[Literal["alpaca", "mistral"], str] = "alpaca",
) -> str:
    """Provided with a task instruction and (optionally) supplementary input, fill them into a QA template and return the resulting prompt.

    :param instruction: The task instruction.
    :type instruction: str
    :param input: Supplementary input to the task, defaults to "".
    :type input: str, optional
    :param suffix: Suffix to add to the prompt, defaults to "".
    :type suffix: str, optional
    :param model_repoid_or_path: The model repo ID or path (e.g., "meta-llama/Llama-3.1-8B-Instruct"), or one of the special values "alpaca" or "mistral", defaults to "alpaca".
    :type model_repoid_or_path: Union[Literal["alpaca", "mistral"], str], optional
    :return: The prompt with the instruction and input filled in.
    :rtype: str
    """

    instruction = instruction.strip()
    input = input.strip()

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
        if suffix:
            warnings.warn(
                f"Suffix not supported except with mistral template. Ignoring suffix."
            )

        prompt = dict_to_dialogue_list({"instruction": instruction, "input": input})
        tokenizer = AutoTokenizer.from_pretrained(model_repoid_or_path)
        input_full = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(input_full, str)

    return input_full
