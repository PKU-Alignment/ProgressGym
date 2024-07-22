from src.abstractions import Model, Data, DataFileCollection
from src.text_writer import write_log
from collections import defaultdict
from typing import List
import os
import json
import time
import random
import re
from tqdm import tqdm


def get_directory_size_bytes(f: str) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(f):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total


def generate_hyperparam(attempt_id: int = None) -> tuple:

    # when attempt_id is None, return the optimal hyperparam
    # deepspeed --num_gpus 8 --master_port=9902 ./libs/llama_factory/src/train_bash.py --deepspeed ./src/abstractions/configs/LF_examples/full_multi_gpu/ds_z3_config.json --ddp_timeout 180000000 --stage pt --do_train --do_sample --model_name_or_path /mnt/models-pku/progressalign/shared_storage/downloaded_models/llama3-8b-base --dataset C013_data --dataset_dir ./libs/llama_factory/data --template alpaca --finetuning_type full --lora_target q_proj,v_proj --output_dir ./output/training_results/HPTuning_C013_llama3-8b-base_20240425_200243_1th/ --overwrite_cache --overwrite_output_dir --cutoff_len 1024 --preprocessing_num_workers 16 --per_device_train_batch_size 8 --per_device_eval_batch_size 16 --gradient_accumulation_steps 1 --lr_scheduler_type polynomial --lr_scheduler_kwargs "{\"lr_end\": 5e-08, \"power\": 22.627416997969522}" --logging_steps 5 --save_steps 500 --warmup_steps 20 --eval_steps 1 --learning_rate 0.0000150000 --save_total_limit 4 --save_strategy steps --logging_first_step --evaluation_strategy steps --val_size 0.1 --dpo_ftx 1.0 --num_train_epochs 7 --plot_loss --fp16
    if attempt_id is None:
        # return (7, -2, 1.5e-5, 'polynomial', {'lr_end': 5e-8, 'power': 22.627416997969522})
        return (4, -2, 1.5e-5, "polynomial", {"lr_end": 5e-8, "power": 11})

    if attempt_id == 0:
        return (
            7,
            -2,
            1.63147e-5,
            "linear",
            {"end_lr": 3e-3, "linear_training_steps": 63},
        )
    elif attempt_id == 1:
        return (
            7,
            -2,
            1.63147e-5,
            "linear",
            {"end_lr": 1e-3, "linear_training_steps": 66},
        )
    elif attempt_id == 2:
        return (
            7,
            -2,
            1.63147e-5,
            "linear",
            {"end_lr": 7e-3, "linear_training_steps": 66},
        )
    elif attempt_id == 3:  # Run ID: leafy-leaf-48
        return (15, -2, 8.3e-8, "constant_with_warmup", None)
    elif attempt_id == 4:  # Run ID: honest-donkey   -39
        return (15, -2, 1.5e-7, "constant_with_warmup", None)
    elif (
        attempt_id == 5
    ):  # Run ID: None; compare with constants, including the old ones with gradient_accumulation_steps>1
        return (15, -2, 3e-8, "constant_with_warmup", None)
    elif 6 <= attempt_id <= 22 and attempt_id % 2 == 0:  # Run ID: trim-sponge-50
        powers = [2 ** (3 + 3 * i / 8) for i in range(9)]
        random.shuffle(powers)
        return (
            7,
            -2,
            1.5e-5,
            "polynomial",
            {"lr_end": 5e-8, "power": powers[(attempt_id - 6) // 2]},
        )

    lr_scheduler_candidates = [
        "polynomial",
        "polynomial",
        "polynomial",
        "cosine",
        "cosine",
        "constant_with_warmup",
    ]

    ret = [
        7,  # epochs
        -2,  # grad_accu_multiplier_log2
        5e-7 * (10 ** random.uniform(-2.5, 2.5)),  # lr
        random.choice(lr_scheduler_candidates),  # lr_scheduler
        None,  # lr_scheduler_kwargs
    ]

    if ret[3] == "polynomial":
        # polynomial scheduler needs lr_end and power
        ret[4] = {
            "lr_end": ret[2] * (10 ** random.uniform(-3.5, -0.5)),
            "power": 2 ** random.uniform(-1, 5.5),
        }

    return ret


def hyperparam_tuning(
    base_model: Model, century: DataFileCollection, iters: int, save_name: str
) -> None:
    century_data: Data = century.convert_to_Data(
        century.collection_name + "_data", forced_rewrite=True
    )
    century_data.set_key_fields(prompt_field_name="content")

    batchsize_multiplier_log2 = (
        -4
        if "70b" in base_model.model_name
        else -1 if "13b" in base_model.model_name else 0
    )

    attempts = 0
    while iters > 0:
        try:
            epochs, grad_accu_multiplier_log2, lr, lr_scheduler, lr_scheduler_kwargs = (
                generate_hyperparam(attempts)
            )
            write_log(
                f"({iters} iters left, attempt {attempts}) Starting one iter: epochs={epochs}, grad_accu_multiplier_log2={grad_accu_multiplier_log2}, lr={lr}, lr_scheduler={lr_scheduler}, lr_scheduler_kwargs={lr_scheduler_kwargs}.",
                "hyperparam_tuning",
            )

            attempts += 1
            base_model.finetune(
                century_data,
                "pretrain",
                "full_param",
                f"HPTuning_{save_name}_{attempts}th",
                epochs=epochs,
                batch_size_multiplier_log2=batchsize_multiplier_log2,
                grad_accu_multiplier_log2=grad_accu_multiplier_log2,
                lr=lr,
                lr_scheduler_type=lr_scheduler,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                load_best_at_end=False,
            )

            iters -= 1
            write_log(
                f"({iters} iters left, attempt {attempts}) Completed one iter: epochs={epochs}, grad_accu_multiplier_log2={grad_accu_multiplier_log2}, lr={lr}, lr_scheduler={lr_scheduler}.",
                "hyperparam_tuning",
            )

        except Exception as e:
            write_log(
                f"({iters} iters left, attempt {attempts}) Error in hyperparameter tuning: {type(e)} {e}.",
                "hyperparam_tuning",
            )

            # If e is KeyboardInterrupt, we should stop the training
            if isinstance(e, KeyboardInterrupt) or attempts > iters * 2:
                print("Hyperparameter tuning interrupted.")
                break


def run_training(dataset_dir: str, models_save_dir: str, num_gpus: int = None):
    
    max_dataset_size_MB = 3000 # originally 300

    sub_datasets = sorted(
        [
            f
            for f in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, f))
        ],
        key=lambda f: int(f.strip("C")),
    )

    centuries: List[DataFileCollection] = []
    for sub in sub_datasets:

        try:
            selected_samples = DataFileCollection(f"{sub}_random_sample")
            selected_size_MB = (
                get_directory_size_bytes(selected_samples.collection_path) / 1024 / 1024
            )
            print(f"Reusing stored. {selected_size_MB}MB (expected ~{max_dataset_size_MB}MB).")
            centuries.append(selected_samples)
            continue

        except:
            pass

        directory_size_MB = (
            get_directory_size_bytes(os.path.join(dataset_dir, sub)) / 1024 / 1024
        )
        full_collection = DataFileCollection(
            sub, "pretrain", os.path.join(dataset_dir, sub)
        )

        if directory_size_MB <= max_dataset_size_MB:
            print(f"Loaded {sub} ({directory_size_MB}MB) as is.")
            centuries.append(full_collection)

        else:  # randomly select a portion of samples to reduce size to max_dataset_size_MB
            print(f"Loading {sub} ({directory_size_MB}MB) with random sampling...")
            selected_samples = full_collection.transform(
                lambda sample_list: random.sample(
                    sample_list, int(max_dataset_size_MB / directory_size_MB * len(sample_list) + 0.5)
                ),
                f"{sub}_random_sample",
                forced_rewrite=True,
                max_batch_size=262144,
            )
            selected_size_MB = (
                get_directory_size_bytes(selected_samples.collection_path) / 1024 / 1024
            )
            print(f"Done. Selected {selected_size_MB}MB (expected ~{max_dataset_size_MB}MB).")
            centuries.append(selected_samples)

    # Calculate, store, and display the ratio of different data sources for each century
    century_stats = {}
    for i, century in enumerate(centuries):
        print(f"Processing century {i + 1}...")

        source_stat_coarse = defaultdict(lambda: 0)
        source_stat_fine = defaultdict(lambda: 0)
        lang_stat = defaultdict(lambda: 0)
        for sample_dict in century.all_passages():
            assert (
                "content" in sample_dict and "source_dataset" in sample_dict
            ), "Invalid sample format"
            assert isinstance(
                sample_dict["content"], str
            ), "Dataset content field contains non-str data"
            size = len(sample_dict["content"])
            source = sample_dict["source_dataset"]
            source_stat_coarse[source] += size

            fine_source = ""
            if "source_document" in sample_dict:
                fine_source += sample_dict["source_document"]
            if "source_dataset_detailed" in sample_dict:
                fine_source += sample_dict["source_dataset_detailed"]

            if fine_source:
                source_stat_fine[source + " / " + fine_source] += size

            if "culture" in sample_dict:
                # leave only latin latters in the culture field
                culture_stripped = re.sub(
                    r"[^a-zA-Z]", "", sample_dict["culture"]
                ).lower()
                lang_stat[culture_stripped] += size
            else:
                lang_stat["unknown"] += size

        century_stats[century.collection_name] = {
            "source_stat_coarse": source_stat_coarse,
            "source_stat_fine": source_stat_fine,
            "lang_stat": lang_stat,
        }

    os.makedirs("./logs", exist_ok=True)
    with open("./logs/century_stats.json", "w") as f:
        json.dump(century_stats, f)

    # Start training
    print("Starting training...")
    base_model_list = [
        Model('Meta-Llama-3-8B', False, num_gpus=num_gpus),
        # Model('Llama-2-7b-hf', False, num_gpus=num_gpus),
        # Model('Llama-2-13b-hf', False, num_gpus=num_gpus),
        # Model("Meta-Llama-3-70B", False, num_gpus=num_gpus),
        # Model('Llama-2-70b-hf', False, num_gpus=num_gpus)
    ]

    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    os.makedirs(models_save_dir, exist_ok=True)

    try:
        instruct_dataset = Data("curated_instructions_3k")
    except:
        instruct_dataset = Data("timeless_qa")
    instruct_dataset.set_key_fields(
        prompt_field_name="instruction",
        query_field_name="input",
        response_field_name="output",
    )

    # If the env var HYPERPARAM_TUNING is set, we will executing hyperparameter tuning
    if (
        os.getenv("HYPERPARAM_TUNING")
        and "true" in os.getenv("HYPERPARAM_TUNING").lower()
    ):
        hyperparam_tuning(
            base_model_list[0],
            centuries[0],
            iters=30,
            save_name=f"{centuries[0].collection_name}_{base_model_list[0].model_name}_{time_stamp}",
        )
        hyperparam_tuning(
            base_model_list[0],
            centuries[-1],
            iters=5,
            save_name=f"{centuries[-1].collection_name}_{base_model_list[0].model_name}_{time_stamp}",
        )

    for base_model in base_model_list:
        batchsize_multiplier_log2 = (
            -2
            if "70b" in base_model.model_name.lower()
            else -1 if "13b" in base_model.model_name.lower() else 0
        )
        num_nodes = (
            4
            if "70b" in base_model.model_name.lower()
            else 2 if "13b" in base_model.model_name.lower() else 1
        )

        sft_todolist = []

        for century in centuries:

            century_data: Data = century.convert_to_Data(
                century.collection_name + "_data",
                forced_rewrite=True,
                filter_fields=["content"],
            )
            century_data.set_key_fields(prompt_field_name="content")

            for sample_dict in tqdm(century_data.all_passages()):
                assert isinstance(
                    sample_dict["content"], str
                ), "Dataset content field contains non-str values."
                for key, value in sample_dict.items():
                    if not (
                        value is None
                        or isinstance(value, str)
                        or isinstance(value, int)
                    ):
                        print(key, value, sample_dict)
                        assert False, "Non-simple data types present in dataset."

            num_epochs = (
                4
                if (
                    "C013" in century.collection_name
                    or "C014" in century.collection_name
                    or "70b" not in base_model.model_name.lower()
                )
                else 3 if ("C015" in century.collection_name) else 1
            )

            tuned_model = base_model.finetune(
                century_data,
                "pretrain",
                "full_param",
                f"{century.collection_name}_{base_model.model_name}_pretrain_{time_stamp}",
                epochs=num_epochs,
                batch_size_multiplier_log2=batchsize_multiplier_log2,
                lr=3e-6,
                num_nodes=num_nodes,
                save_checkpoints=True,
                perform_eval=True,
            )
            tuned_model.save_permanent(os.path.join(models_save_dir, f'{century.collection_name}_{base_model.model_name}_pretrain'), forced_rewrite=True)

            sft_todolist.append((tuned_model, century))

        for tuned_model, century in sft_todolist:
            instruct_model = tuned_model.finetune(
                instruct_dataset,
                "sft",
                "full_param",
                f"{century.collection_name}_{base_model.model_name}_instruct_{time_stamp}",
                epochs=1,
                batch_size_multiplier_log2=batchsize_multiplier_log2,
                lr=3e-6,
                num_nodes=num_nodes,
                save_checkpoints=False,
                perform_eval=False,
            )
            
            instruct_model.save_permanent(os.path.join(models_save_dir, f'{century.collection_name}_{base_model.model_name}_instruct'), forced_rewrite=True)