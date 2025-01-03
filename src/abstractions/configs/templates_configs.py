from src.path import root
from string import Template
import json
import os, sys
from typing import Dict, Any, Literal, Optional, List, Union, Callable


class GlobalState:

    # Public variables
    continuous_backend: bool = False

    # Private variables
    __active_backend_destroyers: List[Callable[[], None]] = []

    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Temporarily set global state variables of ProgressGym for the duration of a context manager block.

        Example:
        ```
        with GlobalState(continuous_backend=True):
            # code that requires continuous backends (i.e. backend does not die after each call to inference, and it kept alive for reuse)
        ```
        """
        self.kwargs = kwargs

    def __enter__(self):
        self.prior_state = {k: getattr(GlobalState, k) for k in self.kwargs.keys()}
        for k, v in self.kwargs.items():
            setattr(GlobalState, k, v)

    def __exit__(self, type, value, traceback):
        for k, v in self.prior_state.items():
            setattr(GlobalState, k, v)

        # Keep silent when destroying backends if LOUD_BACKEND is not set
        old_stdout, old_stderr = sys.stdout, sys.stderr
        with open(os.devnull, "w") as devnull:
            if not eval(os.environ.get("LOUD_BACKEND", "False")):
                sys.stdout = devnull
                sys.stderr = devnull
        
            for destroy in GlobalState.__active_backend_destroyers:
                destroy()
            
            sys.stdout, sys.stderr = old_stdout, old_stderr

        GlobalState.__active_backend_destroyers = []

    def register_destroyer(destroyer: Callable[[], None]):
        GlobalState.__active_backend_destroyers.append(destroyer)


bash_command_template = f"""PYTHONNOUSERSITE=1 MASTER_PORT=9902 conda run --no-capture-output -n %s deepspeed %s --master_port=9902 {root}/libs/llama_factory/src/train_bash.py \\
    --deepspeed %s \\
    --ddp_timeout 180000000 \\
    --stage %s \\
    --do_%s \\%s
    --model_name_or_path %s \\
    --dataset %s \\
    --dataset_dir {root}/libs/llama_factory/data \\%s
    --finetuning_type %s \\
    --lora_target q_proj,v_proj \\
    --output_dir %s \\
    --overwrite_cache \\
    --overwrite_output_dir \\
    --cutoff_len 1024 \\
    --preprocessing_num_workers 16 \\
    --per_device_train_batch_size %d \\
    --per_device_eval_batch_size %d \\
    --gradient_accumulation_steps %d \\
    --lr_scheduler_type %s \\%s
    --logging_steps %.10f \\
    --save_steps %.10f \\
    --warmup_ratio %.10f \\
    --eval_steps %.10f \\
    --learning_rate %.10f \\
    --save_total_limit 4 \\
    --save_strategy %s \\
    --logging_first_step \\
    --evaluation_strategy %s \\%s
    --val_size 0.1 \\
    --dpo_ftx 0.04 \\
    --num_train_epochs %.10f \\%s
    --plot_loss \\%s
    --fp16"""

bash_command_for_ppo = f"""PYTHONNOUSERSITE=1 MASTER_PORT=9902 conda run --no-capture-output -n %s deepspeed %s --master_port=9902 {root}/libs/llama_factory/src/train_bash.py \\
    --deepspeed %s \\
    --model_name_or_path %s \\
    --reward_model %s \\
    --reward_model_type full \\
    --ddp_timeout 180000000 \\
    --stage ppo \\
    --do_train \\
    --finetuning_type %s \\
    --lora_target q_proj,v_proj \\
    --dataset %s \\
    --dataset_dir {root}/libs/llama_factory/data \\%s
    --cutoff_len 1024 \\
    --overwrite_cache \\
    --preprocessing_num_workers 16 \\
    --output_dir %s \\
    --plot_loss \\
    --overwrite_output_dir \\
    --per_device_train_batch_size %d \\
    --per_device_eval_batch_size %d \\
    --gradient_accumulation_steps %d \\
    --learning_rate 1e-6 \\
    --num_train_epochs %.10f \\
    --lr_scheduler_type polynomial \\
    --logging_steps 1 \\
    --save_steps 18 \\
    --report_to wandb \\
    --eval_steps 16 \\
    --evaluation_strategy steps \\
    --save_strategy steps \\
    --warmup_ratio 0.075 \\
    --fp16 \\
    --max_new_tokens 1024 \\
    --do_sample True \\
    --top_p 0.9"""

bash_command_for_rw = f"""PYTHONNOUSERSITE=1 MASTER_PORT=9902 conda run --no-capture-output -n %s deepspeed %s --master_port=9902 {root}/libs/llama_factory/src/train_bash.py \\
    --deepspeed %s \\
    --stage rm \\
    --do_train \\
    --model_name_or_path %s \\
    --finetuning_type full \\
    --dataset %s \\
    --dataset_dir {root}/libs/llama_factory/data \\%s
    --lora_target q_proj,v_proj \\
    --output_dir %s \\
    --overwrite_cache \\
    --overwrite_output_dir \\
    --cutoff_len 1024 \\
    --preprocessing_num_workers 16 \\
    --per_device_train_batch_size %d \\
    --per_device_eval_batch_size %d \\
    --gradient_accumulation_steps %d \\
    --lr_scheduler_type %s \\%s
    --logging_steps 80 \\
    --warmup_ratio 0.075 \\
    --save_steps 10000 \\
    --eval_steps 0.2 \\
    --evaluation_strategy steps \\
    --learning_rate 4e-7 \\
    --num_train_epochs %.10f \\
    --val_size 0.1 \\
    --plot_loss \\
    --fp16"""

bash_command_for_lora_merging = f"""PYTHONNOUSERSITE=1 conda run --no-capture-output -n %s python {root}/libs/llama_factory/src/export_model.py \\
    --model_name_or_path %s \\
    --adapter_name_or_path %s \\%s
    --finetuning_type lora \\
    --export_dir %s \\
    --export_size 2 \\
    --export_legacy_format False"""

eval_command = Template(
    """python3 -m libs.moralchoice.src.evaluate --experiment-name ${name}_single --dataset "low" --model ${name} --question-types "ab" "repeat" "compare"  --eval-nb-samples 5 --add-path ${dir} ;
    python3 -m libs.moralchoice.src.collect --experiment-name ${name}_single  --dataset "low" ;
    python3 -m libs.moralchoice.src.evaluate --experiment-name ${name}_single --dataset "high" --model ${name} --question-types "ab" "repeat" "compare"  --eval-nb-samples 5 --add-path ${dir} ;
    python3 -m libs.moralchoice.src.collect --experiment-name ${name}_single  --dataset "high" ;
"""
)

with open(
    f"{root}/src/abstractions/configs/abstractions_config.json", "r"
) as config_file:
    abstractions_config = json.load(config_file)

    data_search_paths: List[str] = abstractions_config["data_search_paths"]
    data_save_path: str = abstractions_config["data_save_path"]

    if not os.path.exists(data_save_path):
        data_save_path = f"{root}/" + data_save_path
    if not os.path.exists(data_save_path):
        print(f"Data save path {data_save_path} doesn't exist. Creating it.")
        os.makedirs(data_save_path)

    for i, path in enumerate(data_search_paths):
        if not os.path.exists(path):
            data_search_paths[i] = f"{root}/" + path

    model_search_paths: List[str] = abstractions_config["model_search_paths"]
    model_save_path: str = abstractions_config["model_save_path"]

    if not os.path.exists(model_save_path):
        model_save_path = f"{root}/" + model_save_path
    if not os.path.exists(model_save_path):
        print(f"Model save path {model_save_path} doesn't exist. Creating it.")
        os.makedirs(model_save_path)

    for i, path in enumerate(model_search_paths):
        if not os.path.exists(path):
            model_search_paths[i] = f"{root}/" + path

    multinode_master_addr: str = abstractions_config["multinode_master_addr"]
