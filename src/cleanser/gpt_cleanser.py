from typing import List, Tuple, Dict, Optional, Set
from src.abstractions import DataFileCollection
from src.gpt_utils import (
    context_len,
    gpt,
    convo_get_response,
    convo_clear_history,
    independent_get_response,
)
from src.text_writer import write_log
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re

MAX_PARALLEL = 32

check_template = """Given a piece of historical text data, check if the problems listed below exist and are extremely serious in the text.
Problems to look for:                                                      
1. The text is extremely messy or unreadable.
2. The text contains many OCR errors. 
3. The text is written in hard-to-comprehend ancient English, or languages other than English.
4. There are long editor introductions, notes, logistics information, publication information, or other content added by modern editors that obviously do not belong to the original text.

Text: \"\"\"
%s
\"\"\"
"""

clean_template = """Clean the following piece of historical text data. 
Requirements:                                                      
1. Make the text clean and perfectly readable, while sticking to the original content as much as possible. 
2. Remove all line breaks, whitespaces, or other meaningless characters unless they are really necessary.
3. Remove meaningless or completely unreadable content.
4. Remove introductions, notes, logistics information, publication information, or other content added by modern editors that obviously do not belong to the original text.
5. Translate ancient English or non-English languages into modern English. Be as faithfulness as possible to the original content.
6. Correct OCR errors if and when they occur.
7. Output the cleaned text only, with no other comments or any kind of added prefix/suffix.

Text: \"\"\"
%s
\"\"\"
"""


def segment(text: str, max_len: int = context_len) -> Tuple[List[str], str]:
    """Break up line text into segments, ideally at linebreaks or periods.
    This is meant to fit text into context window.
    Returns Tuple[segments, separator], for example (['line 1', 'line 2'], '\\n').
    max_len is given in chars."""

    if len(text) <= max_len:
        return ([text], "")

    # Try line break first, if fails then consider periods. If fails again then consider commas. Fails again then consider spaces.
    for separator, separator_escaped in [
        ("\n", "\n"),
        (".", "\."),
        (",", ","),
        (" ", " "),
    ]:
        text_dedup = re.sub(f"{separator_escaped}{separator_escaped}+", separator, text)
        units = text_dedup.split(separator)
        if max(len(s) for s in units) <= max_len:
            units = units[::-1]
            segments = [""]
            while len(units):
                if len(segments[-1]) + len(units[-1]) > max_len:
                    segments.append("")

                if segments[-1]:
                    segments[-1] += separator
                segments[-1] += units[-1]
                units.pop()

            return (segments, separator)

    # No separator works, just separate with brute force
    return ([text[i : i + max_len] for i in range(0, len(text), max_len)], "")


all_source_dataset = ["EEBO", "gutenberg", "Internet_Archive", "Pile_of_Law"]
messy_source_dataset = ["EEBO", "Internet_Archive"]
chars_count: Dict[str, int] = {
    dataset: 0 for dataset in all_source_dataset
}  # maps from dataset name to total length (in chars)
messy_chars_count = 0  # the total length of documents to be cleaned
cleaned_chars_count = 0  # the total length of currently-cleaned documents
next_milestone = 0.0001
success_count, fail_count, fatal_count = (0, 0, 0)

logger_lock = Lock()


def clean(sample_dict: dict) -> dict:
    """Clean a data sample by feeding the content field to GPT."""

    if (
        "source_dataset" not in sample_dict
        or sample_dict["source_dataset"] not in all_source_dataset
    ):
        logger_lock.acquire()
        write_log(
            f'Cleaning: source_dataset field not found or unrecognized. Source dataset: {"[NA]" if "source_dataset" not in sample_dict else sample_dict["source_dataset"]}. Available fields: {list(sample_dict.keys())}.'
        )
        logger_lock.release()
        return sample_dict

    # if the dataset is a clean one, no need to go through GPT.
    if sample_dict["source_dataset"] not in messy_source_dataset:
        return sample_dict

    if "content" not in sample_dict:
        logger_lock.acquire()
        write_log(
            f"Cleaning: content field not found. Available fields: {list(sample_dict.keys())}."
        )
        logger_lock.release()
        return sample_dict

    segments, separator = segment(sample_dict["content"])
    cleaned_segments = []

    for seg in segments:
        global success_count, fail_count
        for _ in range(5):
            try:
                if len(seg.strip()) < 7:
                    cleaned_segments.append(seg)
                    continue

                cleaned_seg = independent_get_response(clean_template % (seg,))
                cleaned_segments.append(cleaned_seg)
                success_count += 1
                break
            except Exception as e:
                fail_count += 1
                logger_lock.acquire()
                write_log(
                    f"Cleaning segment failed: {type(e)} {e}. Current failure ratio {fail_count}/{fail_count + success_count}. Retrying."
                )
                logger_lock.release()
                time.sleep(1)

    global cleaned_chars_count
    cleaned_chars_count += len(sample_dict["content"])
    sample_dict["content"] = separator.join(cleaned_segments)

    return sample_dict


def clean_parallel(sample_dicts: List[dict]) -> List[dict]:
    """Clean a data sample by feeding the content field to GPT.
    Parallelized to utilize GPT API as much as possible."""

    subtask_args: List[Tuple[Tuple[int, int], str]] = []
    separators: Dict[int, str] = {}
    dict_results: Dict[int, Dict[int, Dict]] = {i: {} for i in range(len(sample_dicts))}
    predetermined: Set[int] = set()

    for dict_id, sample_dict in enumerate(sample_dicts):
        if (
            "source_dataset" not in sample_dict
            or sample_dict["source_dataset"] not in all_source_dataset
        ):
            write_log(
                f'Cleaning: source_dataset field not found or unrecognized. Source dataset: {"[NA]" if "source_dataset" not in sample_dict else sample_dict["source_dataset"]}. Available fields: {list(sample_dict.keys())}.'
            )
            predetermined.add(dict_id)
            continue

        # if the dataset is a clean one, no need to go through GPT.
        if sample_dict["source_dataset"] not in messy_source_dataset:
            predetermined.add(dict_id)
            continue

        if "content" not in sample_dict:
            write_log(
                f"Cleaning: content field not found. Available fields: {list(sample_dict.keys())}."
            )
            predetermined.add(dict_id)
            continue

        segments, separator = segment(sample_dict["content"])
        separators[dict_id] = separator

        for seg_id, seg in enumerate(segments):
            subtask_args.append(((dict_id, seg_id), seg))

    def subtask_handler(
        index: Tuple[int, int], seg: str
    ) -> Tuple[Tuple[int, int], Optional[str]]:
        global success_count, fail_count, cleaned_chars_count, fatal_count
        sleep_time = 2
        for _ in range(5):
            try:
                if len(seg.strip()) < 7:
                    return index, seg

                cleaned_seg = independent_get_response(clean_template % (seg,))
                success_count += 1
                cleaned_chars_count += len(seg) + 1
                return index, cleaned_seg
            except Exception as e:
                logger_lock.acquire()
                fail_count += 1
                write_log(
                    f"Cleaning segment failed: {type(e)} {e}. Current failure-to-success ratio {fail_count}/{success_count}. Retrying."
                )
                logger_lock.release()
                time.sleep(sleep_time)
                sleep_time *= 3

        # failed for 5 times in a row
        logger_lock.acquire()
        fatal_count += 1
        write_log(
            f"FATAL - Cleaning segment failed for 5 times. Current fatal-to-success ratio {fatal_count}/{success_count}."
        )
        logger_lock.release()
        return index, None

    # Initialize ThreadPoolExecutor with the desired number of workers
    # If not specified, it defaults to the number of processors on the machine, multiplied by 5
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        # Submit all tasks to the executor and receive futures
        futures = [executor.submit(subtask_handler, *args) for args in subtask_args]

        # as_completed(futures) returns an iterator over Future instances as they complete (are done)
        for future in as_completed(futures):
            index, cleaned_seg = future.result()  # Retrieve the result from the future
            if cleaned_seg is not None:
                dict_results[index[0]][
                    index[1]
                ] = cleaned_seg  # Place response in the correct order based on its index

            global next_milestone
            current_completion = cleaned_chars_count / max(1, messy_chars_count) * 100
            if current_completion >= next_milestone:
                next_milestone *= 1.1
                logger_lock.acquire()
                write_log(
                    f"Cleaning: Completed {current_completion}% = {cleaned_chars_count}/{messy_chars_count}, next milestone {next_milestone}%."
                )
                logger_lock.release()

    for dict_id, sample_dict in enumerate(sample_dicts):
        if dict_id in predetermined:
            continue

        cleaned_segments = (
            [dict_results[dict_id][i] for i in sorted(dict_results[dict_id])]
            if dict_id in dict_results
            else []
        )
        new_content = separator.join(cleaned_segments)

        if len(new_content) < 15:
            global fail_count
            sample_dict["content"] = sample_dict["content"][:100]
            fail_count += 1
            write_log(
                f'Cleaning: one sample becomes near empty ("{new_content}") after cleaning. Current failure ratio {fail_count}/{fail_count + success_count}. Original sample_dict (content truncated) = {sample_dict}.'
            )

        sample_dict["content"] = new_content

    return sample_dicts


if __name__ == "__main__":

    # histext = DataFileCollection(
    #     collection_name = 'histext_all',
    #     is_instruction_data = False,
    #     collection_path = '../../shared_storage/our_datasets/HisText_Mar8_Guten_EEBO_PoL_IA10_unrefined/',
    #     file_selection_func = (lambda path: 'Y' in path)
    # )

    # chars_count = {'EEBO': 7822164233, 'gutenberg': 1052629906, 'Internet_Archive': 5221473604, 'Pile_of_Law': 35669322256}

    histext = DataFileCollection(
        collection_name="histext_every_100yr_or_pre15c",
        is_instruction_data=False,
        collection_path="../../shared_storage/our_datasets/HisText_Mar8_Guten_EEBO_PoL_IA10_unrefined/",
        file_selection_func=(
            lambda path: "Y" in path
            and (
                int(path.split("/")[-1][1:6]) % 100 == 0
                or int(path.split("/")[-1][1:6]) < 1400
            )
        ),
    )

    # calculate chars_count and messy_chars_count
    for sample_dict in histext.all_passages():
        if (
            "source_dataset" not in sample_dict
            or sample_dict["source_dataset"] not in all_source_dataset
        ):
            write_log(
                f'Pre-Cleaning: source_dataset field not found or unrecognized. Source dataset: {"[NA]" if "source_dataset" not in sample_dict else sample_dict["source_dataset"]}. Available fields: {list(sample_dict.keys())}.'
            )
            continue

        if "content" not in sample_dict:
            write_log(
                f"Pre-Cleaning: content field not found. Available fields: {list(sample_dict.keys())}."
            )
            continue

        chars_count[sample_dict["source_dataset"]] += len(sample_dict["content"])

    messy_chars_count = sum(chars_count[dataset] for dataset in messy_source_dataset)

    print(f"messy_chars_count = {messy_chars_count}\nchars_count = {chars_count}\n")
    write_log(f"messy_chars_count = {messy_chars_count}\nchars_count = {chars_count}\n")

    histext_cleaned = histext.transform(
        transformation=clean_parallel,
        # result_collection_name='histext_all_cleaned',
        result_collection_name="histext_every_100yr_or_pre15c_cleaned",
        max_batch_size=MAX_PARALLEL,
    )
