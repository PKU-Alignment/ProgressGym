from threading import Thread
from typing import List, Tuple
import os
import openai
import guidance
import logging
import json
import re
import csv

if not os.path.exists("./logs/eval"):
    os.makedirs("./logs/eval")

with open("./src/abstractions/configs/abstractions_config.json", "r") as config_file:
    abstractions_config = json.load(config_file)
    if "openai_mirror" in abstractions_config:
        mirror_url = abstractions_config["openai_mirror"]
    else:
        mirror_url = input("OpenAI Mirror URL: ").strip() or "https://api.openai.com/v1"

    if "openai_key" in abstractions_config:
        api_key = abstractions_config["openai_key"]
    else:
        api_key = input("OpenAI API Key: ").strip()

os.environ["OPENAI_BASE_URL"] = mirror_url
os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_LOG"] = "info"

model_version = input("Use GPT-4 or GPT-3.5? (write 4 or 3.5) ")
model_name, context_len = (
    ("gpt-4o", 120000) if "4" in model_version else ("gpt-3.5-turbo-0125", 15000)
)
print(f"Model name {model_name}, Context length {context_len}.")

# set up guidance
gpt = guidance.models.OpenAI(model_name)

# set up OpenAI
openai_client = openai.OpenAI()
system_prompt = f"You are a research-oriented large language model trained by OpenAI, based on the GPT-4 architecture."
global_message_history = [
    {"role": "system", "content": system_prompt}
]  # the main conversation


def get_system_prompt(src="default"):
    prompt_dir = os.path.join("src", "evaluation", "assets", "expand_prompt.json")
    if src == "default":
        return f"You are a research-oriented large language model trained by OpenAI, based on the GPT-4 architecture."
    with open(prompt_dir, "r") as f:
        boy = json.load(f)
    return boy["system"] + boy["info_" + src]


system_prompt = get_system_prompt("foundation")


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


def convo_get_response(prompt: str) -> str:
    """Within the main conversation, ask `prompt` and returns model answer."""
    # add prompt to history
    global_message_history.append({"role": "user", "content": prompt})

    chat_completion = openai_client.chat.completions.create(
        model=model_name, messages=global_message_history
    )
    # add response to history
    response = dict(dict(chat_completion.choices[0])["message"])["content"]
    global_message_history.append({"role": "assistant", "content": response})
    return response


def convo_clear_history() -> None:
    """Clear conversation history of the main conversation."""
    global global_message_history
    global_message_history = [
        {"role": "system", "content": system_prompt}
    ]  # the main conversation


def independent_get_response(prompt: str, temperature: float = 0.2) -> str:
    """In a new independent conversation, ask `prompt` and returns model answer"""

    chat_completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return dict(dict(chat_completion.choices[0])["message"])["content"]


def __independent_get_response_parallel_helper(
    index: int, prompt: str, responses: dict
) -> None:
    """In a new independent conversation, ask `prompt` and returns model answer. Helper function for the parallelized version of this function."""

    chat_completion = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    response = dict(dict(chat_completion.choices[0])["message"])["content"]
    responses[index] = response


def independent_get_response_parallel(prompts: List[str]) -> List[str]:
    """In independent conversations, ask the questions in `prompts` separately and returns the respective model answers.
    Parallelized with multithreading. Number of threads equal the size of `prompts`."""

    threads = []
    responses = {}

    # Using enumerate to keep track of each prompt's position
    for index, prompt in enumerate(prompts):
        args = (index, prompt, responses)
        thread = Thread(target=__independent_get_response_parallel_helper, args=args)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Sorting responses by the index and returning only the response in the correct order
    ordered_responses = [responses[index] for index in sorted(responses)]
    return ordered_responses


def construct_input_for_expand(source):
    proto_dir = os.path.join(
        "src", "evaluation", "raw_dataset", source, "prototype.csv"
    )
    template_dir = "src/evaluation/assets/expand_prompt.json"
    out_dir = os.path.join(os.path.basename(proto_dir), "generated.csv")
    with open(template_dir, "r") as f:
        boi = json.load(f)
        input_template = boi["instruction_" + source] + boi["input"]
    instruction = []
    with open(proto_dir, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            instruction.append(
                input_template.format(
                    theme=source, context=row[2], num=2, example=",".join(row)
                )
            )
    return instruction, header


def _debug(output):
    mode = "a+" if os.path.exists("logs/eval/log.txt") else "w"
    with open("logs/eval/log.txt", mode) as f:
        f.write("\n".join(output) + "\n")


def conduct_expand(source, outdir, bs=8):
    instruction, header = construct_input_for_expand(source)
    answer = []
    for bn in range(len(instruction) // bs):
        end = min(bn * bs + bs, len(instruction))
        instruction_this = instruction[bn * bs : end]
        # _debug(instruction_this)
        answer_this = independent_get_response_parallel(instruction_this)
        # _debug(answer_this)
        answer.extend(answer_this)

    with open(outdir, mode="w") as file:
        file.write(",".join(header) + "\n")
        for row in answer:
            file.write("".join(row) + "\n")
