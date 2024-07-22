"""
We now have access to the following instruction datasets in ../../shared_storage/downloaded_datasets/:

```
root@gpu01:/mnt/models-pku/progressalign/shared_storage/downloaded_datasets# ls
alpaca_gpt4_data_en.json  databricks-dolly-15k.jsonl  hh-rlhf  lima.json  natural-instructions  stanford-SHP
```

We will curate a more value-neutral instruct dataset by doing the following steps:
1. Convert all jsonl files to json files, since DataFileCollection only supports json files
2. Using the DataFileCollection class to load all the instruction dataset under `downloaded_datasets` at once
3. Use the DataFileCollection.transform method to filter out almost all the samples, except ones that are both
            (a) completely value-neutral, AND 
            (b) timeless (i.e. is not specific to a certain time period like the modern era, and can be a reasonable instruction for any time period in the past millennium).
        - The filtering should be implemented with multithreaded calls to GPT, with automatic retries on failure. Long instructions that exceed the context length should be skipped.
        - The transformation should also unify the format of all datasets to the format of alpaca_gpt4_data_en.json, with (optionally) extra metadata fields when needed.
4. Save the curated instruct dataset to a new directory using DataFileCollection.save_permament 

Supplementary information:

#### Structure of alpaca_gpt4_data_en.json

```json
[
  {
    "instruction": "Give three tips for staying healthy.",
    "input": "",
    "output": "1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.\n\n2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.\n\n3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night."
  },
```

#### Structure of databricks-dolly-15k.jsonl

```jsonl
{"instruction": "Which is a species of fish? Tope or Rope", "context": "", "response": "Tope", "category": "classification"}
{"instruction": "Alice's parents have three daughters: Amy, Jessy, and what\u2019s the name of the third daughter?", "context": "", "response": "The name of the third daughter is Alice", "category": "open_qa"}
```

#### Structure of lima.json

```json
[
  {
    "instruction": "Can brain cells move? By movement I mean long distance migration (preferably within the brain only).",
    "input": "",
    "output": "The question is relatively broad and one should [truncated for easier display].",
    "history": []
  },
```

Here, `history` stands for the previous conversation history, which should be kept as is.

#### Structure of hh-rlhf

You should ignore hh-rlhf for now, and don't include them in the dataset.

#### Structure of stanford-SHP

You should ignore stanford-SHP for now, and don't include them in the dataset.

#### Structure of natural-instructions  

You should ignore natural-instructions for now, and don't include them in the dataset.

#### DataFileCollection.transform

```python
def transform(self, transformation: Union[Callable[[Dict], Dict], Callable[[List[Dict]], List[Dict]]], 
                        result_collection_name: str, forced_rewrite: bool = False, max_batch_size: int = 1,
                        suppress_tqdm: bool = False):
        '''
        Apply transformation to every element of the current dataset (in the format of a json list of json dicts where the values are of mutable or immutable types), and returns a DataFileCollection instance containing the resulting dataset.
        Out-of-place. Does not modify self.
        Do not include path in result_collection_name.
        This function (like all others in abstractions) is memory-efficient for huge json files.
        All data files should be json files with the type of List[Dict[Hashable, Any]].
        If max_batch_size is specified and is >1, the transformation function must take inputs of type List[Dict] and return a List[Dict].
        '''
```

#### DataFileCollection.__init__

```python
def __init__(self, collection_name: str, data_type: Literal['pretrain', 'sft', 'preference'] = 'pretrain', 
                       collection_path: Optional[str] = None, file_selection_func: Optional[Callable[[str], bool]] = None, **kwargs):
        '''
        Example: DataFileCollection(collection_name='histext_1826_to_2018', 
                                    data_type='pretrain', 
                                    collection_path = '../../shared_storage/our_datasets/HisText_Mar8_Guten_EEBO_PoL_IA10_unrefined/', 
                                    file_selection_func = (lambda path: 1826 <= int(path.split('/')[-1][1:6]) <= 2018))
        If collection_path is omitted, we will search for collection_name in directories specified in abstractions_config.json.
        When file_selection_func is supplied, files will be captured real-time, instead of only when initializing. Only json files will be captured.
        '''
```
"""

from src.abstractions import Data, DataFileCollection, fill_in_QA_template
from src.text_writer import write_log, read_json_memory_efficient

# Multithreading for GPT interaction
import threading
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.gpt_utils import context_len, independent_get_response

# For converting jsonl to json
import json
import os
import random

# For filtering out non-value-neutral instructions
from typing import List, Dict, Callable
from tqdm import tqdm
import time


def utf8len(s):
    return len(s.encode("utf-8"))


def convert_jsonl_to_json(jsonl_path: str, json_path: str):
    with open(jsonl_path, "r") as f:
        json_data = [json.loads(line) for line in f]

    with open(json_path, "w") as f:
        json.dump(json_data, f)


GPT_prompt_template = """Given as input a Q&A conversation between a human and an AI assistant, determine whether both ...
(1). The human instruction/prompt/question is VALUE-NEUTRAL and TIMELESS, AND
(2). The AI response/answer is VALUE-NEUTRAL and TIMELESS
... are true.
By VALUE-NEUTRAL, we mean that the conversation could have come from anyone with any belief system, and does not contain any subjective opinions, personal beliefs, or culturally-situated statements. 
By TIMELESS, we mean that the conversation could have happened at any time in the past millennium, and does not contain any explicit or implicit reference to specific time periods, events, technologies, or any other time-bound context.

If both (1) and (2) are mostly true, output 'YES' verbatim, otherwise 'NO' verbatim. If you are unsure, output 'SKIP' verbatim. No extra output is allowed.
"""

log_lock = Lock()
rejected_count_lock = Lock()
rejected_count, skipped_count, accepted_count, oversized_count = 0, 0, 0, 0
processed_chars, estimated_total_chars = 0, 0

current_percentage = 0
next_percentage_milestone = 0.001
step_length = 1.15

# Persistent memory storing the retry sleep seconds for each thread
thread_sleep = {}


def filter_instruction(sample_dict: Dict) -> bool:
    global rejected_count, skipped_count, accepted_count, processed_chars, next_percentage_milestone, step_length, current_percentage, estimated_total_chars, oversized_count, thread_sleep

    full_GPT_prompt = fill_in_QA_template(GPT_prompt_template, str(sample_dict))
    if len(full_GPT_prompt) // 2 > context_len:
        log_lock.acquire()
        oversized_count += 1
        write_log(
            f"Skipped {sample_dict} due to excessive length {len(full_GPT_prompt)}.",
            log_name="curate_prompts",
        )
        log_lock.release()
        return False

    id = threading.current_thread().native_id
    assert id is not None, "Thread ID is None"
    retry_sleep_secs = thread_sleep.get(id, random.random() + 0.5) / 1.15
    while retry_sleep_secs < 130:
        time.sleep(retry_sleep_secs)
        try:
            thread_sleep[id] = retry_sleep_secs
            response = independent_get_response(full_GPT_prompt)
            status = "YES" in response[: min(10, len(response))]
            if status:
                log_lock.acquire()
                accepted_count += 1
                current_percentage = processed_chars / estimated_total_chars * 100
                write_log(
                    f'Identified {sample_dict}. Response {response}. Counts: Rejected {rejected_count}, Skipped {skipped_count}, Accepted {accepted_count}, Oversized {oversized_count}. Processed {processed_chars} chars our of estimated {estimated_total_chars}, {"%.5f" % (current_percentage,)}%.',
                    log_name="curate_prompts",
                )
                log_lock.release()

            rejected_count_lock.acquire()
            if not status:
                if "SKIP" in response:
                    skipped_count += 1
                else:
                    rejected_count += 1

            processed_chars += utf8len(repr(sample_dict))
            current_percentage = processed_chars / estimated_total_chars * 100
            rejected_count_lock.release()

            if current_percentage > next_percentage_milestone:
                log_lock.acquire()
                write_log(
                    f"Processed {current_percentage}%, {processed_chars} chars our of estimated {estimated_total_chars}. Counts: Rejected {rejected_count}, Skipped {skipped_count}, Accepted {accepted_count}, Oversized {oversized_count}.",
                    log_name="curate_prompts",
                )
                next_percentage_milestone *= step_length
                log_lock.release()

            return status

        except Exception as e:
            log_lock.acquire()
            retry_sleep_secs *= random.random() * 4 + 2
            write_log(
                f"Error: {type(e)} {e}. Retrying in {retry_sleep_secs} seconds..."
            )
            log_lock.release()

    rejected_count_lock.acquire()
    rejected_count += 1
    processed_chars += utf8len(repr(sample_dict))
    rejected_count_lock.release()

    log_lock.acquire()
    print(f"Failed to get response for {sample_dict}.")
    write_log(f"Failed to get response for {sample_dict}.", log_name="curate_prompts")
    log_lock.release()
    return False


def curate_prompts():

    # Convert all jsonl files to json files
    for file in os.listdir("../../shared_storage/downloaded_datasets/"):
        if file.endswith(".jsonl"):
            print(f"Converting {file} to {file[:-5]}.json...")
            convert_jsonl_to_json(
                f"../../shared_storage/downloaded_datasets/{file}",
                f"../../shared_storage/downloaded_datasets/{file[:-5]}.json",
            )
            os.remove(f"../../shared_storage/downloaded_datasets/{file}")

    # Load all instruction datasets
    instruction_datasets = DataFileCollection(
        "instructions_uncurated_Apr22_alpaca_dolly_lima",
        collection_path="../../shared_storage/downloaded_datasets/",
        file_selection_func=(
            lambda path: "alpaca_gpt4_data_en" in path
            or "databricks-dolly-15k" in path
            or "lima.json" in path
        ),
    )

    # Calculate estimated total chars
    global estimated_total_chars
    estimated_total_chars = 0
    file_count = 0
    for path in instruction_datasets.all_files():
        assert path.endswith(".json"), "Invalid file format"
        assert os.path.exists(path), "File does not exist"
        estimated_total_chars += os.path.getsize(path)
        file_count += 1

    write_log(
        f"Estimated total chars: {estimated_total_chars}. Total files: {file_count}.",
        log_name="curate_prompts",
    )

    def filter_and_unify_format(sample_list: List[Dict]) -> List[Dict]:
        global thread_sleep
        thread_sleep = {}

        # Unify the format of all datasets
        for sample in sample_list:
            if "context" in sample and "input" not in sample:
                sample["input"] = sample["context"]
                del sample["context"]

            if "response" in sample and "output" not in sample:
                sample["output"] = sample["response"]
                del sample["response"]

        # Implement the multithreaded version of the line below
        # filtered_samples = [sample for sample in sample_list if filter_instruction(sample)]

        filtered_samples = []
        lock = Lock()

        def filter_sample(sample):
            if filter_instruction(sample):
                with lock:
                    filtered_samples.append(sample)

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(filter_sample, sample) for sample in sample_list]
            for future in as_completed(futures):
                future.result()

        write_log(
            f"Finished one batch. thread_sleep = {thread_sleep}.",
            log_name="curate_prompts",
        )
        thread_sleep = {}
        return filtered_samples

    curated_dataset = instruction_datasets.transform(
        filter_and_unify_format,
        "instructions_curated_Apr22_alpaca_dolly_lima",
        forced_rewrite=True,
        max_batch_size=4096,
        suppress_tqdm=True,
    )

    # Save the curated instruct dataset
    curated_dataset.save_permanent()
    curated_data = curated_dataset.convert_to_Data(
        "instructions_curated", forced_rewrite=True
    )
    curated_data.set_key_fields(
        prompt_field_name="instruction",
        query_field_name="input",
        response_field_name="output",
    )
    curated_data.save_permanent_and_register()


if __name__ == "__main__":
    curate_prompts()
    print("Finished curating instruct dataset.")
    write_log(f"Finished curating instruct dataset.", log_name="curate_prompts")
