from collections import Counter
import os
from vllm import LLM, SamplingParams
from src.abstractions import Model, Data, fill_in_QA_template, DataFileCollection
from typing import List, Tuple, Dict, Optional, Set
from src.text_writer import write_log, JsonListWriter
import time
import re

MAX_PARALLEL = 262144
context_len = 30000  # for mixtral and mistral-7b

clean_instruction, clean_suffix = (
    """Clean the following piece of historical text, given to you as input. Make the text clean and perfectly readable, while sticking to the original content as much as possible.
If the problems listed below are extremely rampant in the text, output the cleaned text in full without any caveat/comment or added prefix/suffix. Otherwise, simply output "[SKIP]" verbatim, without any explanations, comments, text excerpts, prefix/suffix, or any other output.
Requirements:
1. Remove meaningless or completely unreadable content. Also remove all line breaks, whitespaces, or other meaningless characters unless they are really necessary.
2. Remove introductions, notes, logistics information, publication information, or other content added by modern editors that obviously do not belong to the original text.
3. Translate ancient English or non-English languages into modern English. Be as faithfulness as possible to the original content.
4. Correct OCR errors if and when they occur.""",
    'ONLY OUTPUT THE ENTIRE CLEANED TEXT, with NO other caveats/comments/replies or any kind of added prefix/suffix. Alternatively (if cleaning isn\'t absolutely unnecessary), output "[SKIP]" verbatim, without any explanation, comment, text excerpt, prefix/suffix, or any other output.',
)


def segment(text: str, max_len: int = context_len // 2) -> Tuple[List[str], str]:
    """Break up line text into segments, ideally at linebreaks or periods.
    This is meant to fit text into context window.
    Returns Tuple[segments, separator], for example (['line 1', 'line 2'], '\\n').
    max_len is given in chars."""

    text = re.sub(f"\n\n+", "\n", text)

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
next_milestone = 0.00001
success_count, fail_count, fatal_count = (0, 0, 0)


def clean_parallel(sample_dicts: List[dict]) -> List[dict]:
    """Clean a data sample by feeding the content field to GPT.
    Parallelized to utilize GPT API as much as possible."""

    global next_milestone, cleaned_chars_count, success_count, fatal_count, fail_count

    subtask_args: List[Tuple[Tuple[int, int], str]] = []
    separators: Dict[int, str] = {}
    dict_results: Dict[int, Dict[int, Dict]] = {i: {} for i in range(len(sample_dicts))}
    predetermined: Set[int] = set()

    length_counts = Counter()

    for dict_id, sample_dict in enumerate(sample_dicts):
        if (
            "source_dataset" not in sample_dict
            or sample_dict["source_dataset"] not in all_source_dataset
        ):
            write_log(
                f'Cleaning: source_dataset field not found or unrecognized. Source dataset: {"[NA]" if "source_dataset" not in sample_dict else sample_dict["source_dataset"]}. Available fields: {list(sample_dict.keys())}.',
                "cleanser_full",
            )
            predetermined.add(dict_id)
            continue

        # if the dataset is a clean one, no need to go through GPT.
        if sample_dict["source_dataset"] not in messy_source_dataset:
            predetermined.add(dict_id)
            continue

        if "content" not in sample_dict:
            write_log(
                f"Cleaning: content field not found. Available fields: {list(sample_dict.keys())}.",
                "cleanser_full",
            )
            predetermined.add(dict_id)
            continue

        segments, separator = segment(sample_dict["content"], max_len=1024)
        separators[dict_id] = separator

        for seg_id, seg in enumerate(segments):
            length_counts[len(seg)] += 1
            subtask_args.append(((dict_id, seg_id), seg))

    prompt_template_len = len(
        fill_in_QA_template(clean_instruction, ".", clean_suffix, "mistral")
    )

    write_log(
        f"""Cleaning: dict_count = {len(sample_dicts)}, 
                            seg_count = {len(subtask_args)}, 
                            tot_seg_len = {sum(l*c for l,c in length_counts.items())},
                            prompt_template_len = {prompt_template_len}, 
                            length_counts = {length_counts}""",
        "cleanser_full",
    )

    outputs: List[str] = mixtral_vllm.generate(
        [
            fill_in_QA_template(
                clean_instruction, tp[1].strip(), clean_suffix, "mistral"
            )
            for tp in subtask_args
        ],
        sampling_params,
    )
    assert len(outputs) == len(subtask_args)

    skip_count = 0
    output_length_counts = Counter()
    total_input_length, total_output_length = 0, 0

    cleaned_chars_count += 1
    os.makedirs("./__trash/__trace/", exist_ok=True)
    segs_file, full_file = "./__trash/__trace/segs_%14d.json" % (
        cleaned_chars_count,
    ), "./__trash/__trace/full_%14d.json" % (cleaned_chars_count,)
    with JsonListWriter(segs_file) as data_dump:
        for subtask, output in zip(subtask_args, outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text.strip()
            assert generated_text is not None

            index = subtask[0]
            data_dump.append(
                {"index": index, "prompt": prompt, "generated_text": generated_text}
            )

            success_count += 1
            cleaned_chars_count += len(prompt) - prompt_template_len + 2

            def is_skip(prompt: str, generated_text: str) -> bool:
                if "skip" not in generated_text.lower():
                    return False

                if (
                    "SKIP" in generated_text
                    or "[skip" in generated_text.lower()
                    or "skip]" in generated_text.lower()
                ):
                    return True

                # The remaining cases are ambiguous (it is known that the word skip is present in the response), so we need to check the prompt.
                baseline_skip_counts_in_prompt = clean_instruction.lower().count(
                    "skip"
                ) + clean_suffix.lower().count("skip")

                # If prompt doesn't contain skip (except in the template), then the skip in the response must be a true skip.
                return prompt.lower().count("skip") == baseline_skip_counts_in_prompt

            if not is_skip(prompt, generated_text):
                output_length_counts[len(generated_text)] += 1
                total_input_length += len(prompt) - prompt_template_len + 1
                total_output_length += len(generated_text)

            if len(generated_text) <= 7 or is_skip(prompt, generated_text):
                dict_results[index[0]][index[1]] = subtask[1]
                skip_count += int(is_skip(prompt, generated_text))
            else:
                dict_results[index[0]][
                    index[1]
                ] = generated_text  # Place response in the correct order based on its index

    write_log(
        f"""Cleaning (continued): dict_count = {len(sample_dicts)}, 
                                        skip_rate = {(skip_count+1)/(len(subtask_args)+1)}, 
                                        skip_count = {skip_count}, 
                                        total_count = {len(subtask_args)},
                                        total_input_length = {total_input_length},
                                        total_output_length = {total_output_length},
                                        output_length_counts = {output_length_counts}""",
        "cleanser_full",
    )

    current_completion = cleaned_chars_count / max(1, messy_chars_count) * 100
    if current_completion >= next_milestone:
        next_milestone *= 1.1
        write_log(
            f"Cleaning: Completed {current_completion}% = {cleaned_chars_count}/{messy_chars_count}, next milestone {next_milestone}%.",
            "cleanser",
        )

    with JsonListWriter(full_file) as data_dump:
        for dict_id, sample_dict in enumerate(sample_dicts):
            if dict_id in predetermined:
                continue

            cleaned_segments = (
                [dict_results[dict_id][i] for i in sorted(dict_results[dict_id])]
                if dict_id in dict_results
                else []
            )
            new_content = separator.join(cleaned_segments)

            data_dump.append({**sample_dict, "new_content": new_content})

            if len(new_content) < 15:
                global fail_count
                sample_dict["content"] = sample_dict["content"][:100]
                fail_count += 1
                write_log(
                    f'Cleaning: one sample becomes near empty ("{new_content}") after cleaning. Current failure ratio {fail_count}/{fail_count + success_count}. Original sample_dict (content truncated) = {sample_dict}.',
                    "cleanser_full",
                )

            sample_dict["content"] = new_content

    return sample_dicts


def get_char_counts(histext: DataFileCollection):
    for sample_dict in histext.all_passages():
        if (
            "source_dataset" not in sample_dict
            or sample_dict["source_dataset"] not in all_source_dataset
        ):
            write_log(
                f'Pre-Cleaning: source_dataset field not found or unrecognized. Source dataset: {"[NA]" if "source_dataset" not in sample_dict else sample_dict["source_dataset"]}. Available fields: {list(sample_dict.keys())}.',
                "cleanser_full",
            )
            continue

        if "content" not in sample_dict:
            write_log(
                f"Pre-Cleaning: content field not found. Available fields: {list(sample_dict.keys())}.",
                "cleanser_full",
            )
            continue

        chars_count[sample_dict["source_dataset"]] += len(sample_dict["content"])


def run_cleanser(in_path: str, out_path: str, max_parallel: int = 262144) -> None:

    global sampling_params, mixtral_vllm, MAX_PARALLEL
    MAX_PARALLEL = max_parallel

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0.2, top_p=0.95, max_tokens=1024
    )  # context_len // 2)

    # Create an LLM.
    # mixtral_vllm = LLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1", tensor_parallel_size=8, max_context_len_to_capture = 32000)
    mixtral_vllm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.2", tensor_parallel_size=4
    )  # removed: max_context_len_to_capture = 32000

    histext = DataFileCollection(
        collection_name="histext_all",
        is_instruction_data=False,
        collection_path=in_path,
        file_selection_func=(lambda path: "Y" in path),
    )

    get_char_counts(histext)  # calculate chars_count and messy_chars_count
    # For the dataset before rule-based refinement: chars_count = {'EEBO': 7822164233, 'gutenberg': 1052629906, 'Internet_Archive': 5221473604, 'Pile_of_Law': 35669322256}
    # For the dataset after rule-based refinement: chars_count = {'EEBO': 7788343881, 'gutenberg': 984516417, 'Internet_Archive': 5051273596, 'Pile_of_Law': 26543799682}

    messy_chars_count = sum(chars_count[dataset] for dataset in messy_source_dataset)

    print(f"messy_chars_count = {messy_chars_count}\nchars_count = {chars_count}\n")
    write_log(
        f"messy_chars_count = {messy_chars_count}\nchars_count = {chars_count}\n",
        "cleanser",
    )

    histext_cleaned = histext.transform(
        transformation=clean_parallel,
        result_collection_name="histext_all_cleaned",  # 'histext_every_100yr_or_pre15c_cleaned'
        max_batch_size=MAX_PARALLEL,
        suppress_tqdm=True,
    )

    histext_cleaned.save_permanent(saved_name=out_path, forced_rewrite=True)
    write_log(f"Cleaning: Quit with success.", "cleanser")
