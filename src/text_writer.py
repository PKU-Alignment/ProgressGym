import json
from typing import Tuple, Iterable, Dict, Hashable, Any
import os
import re
from jsonstreamer import JSONStreamer
from time import strftime, localtime

"""
mapping from file id (year number) to number of entries written.
0:  initialized but not written anything yet; 
not in year_lengths:  uninitialized.
"""
year_lengths = {}
os.makedirs("./logs", exist_ok=True)


def write_log(s: str, log_name: str = "build_dataset"):
    time_str = strftime("%d %b %H:%M:%S", localtime())
    with open(f"./logs/{log_name}.log", "a") as log_file:
        log_file.write(time_str + "  ---  " + s + "\n")


def year2path(yr_num: int) -> Tuple[str, str]:
    """
    returns (filefolder, filefullpath) for any given year number
    """
    century = "C%03d" % (yr_num // 100 + 1,)
    century_path = f"./dataset/dataset_text_sequence/{century}/"
    if not os.path.exists(century_path):
        os.mkdir(century_path)

    year = "Y%05d" % yr_num
    year_path = os.path.join(century_path, f"{year}.json")

    return (century_path, year_path)


def initialize_year(yr_num: int):
    """
    initialize a file (a year) by printing '[' into it and setting flag
    """
    if yr_num in year_lengths:
        # already initialized
        return

    path = year2path(yr_num)[1]
    with open(path, "w") as out_file:
        out_file.write("[")

    year_lengths[yr_num] = 0


def write_single_entry(
    json_dict,
    content: str = None,
    creation_year: int = None,
    source_dataset: str = None,
):
    """
    Given an entry (e.g. an article) with specified content, year and source (and optionally other attributes), write it into ./dataset/dataset_text_sequence
    For the mandatory fields (content, creation_year, source_dataset), each them must be specified either in json_dict or in their separate argument, or both. AssertionError will be raised if neither contains that field.
    """

    # check that the necessary fields are indeed filled
    assert type(content) == str or (
        "content" in json_dict and type(json_dict["content"]) == str
    )
    assert type(creation_year) == int or (
        "creation_year" in json_dict and type(json_dict["creation_year"]) == int
    )
    assert type(source_dataset) == str or (
        "source_dataset" in json_dict and type(json_dict["source_dataset"]) == str
    )

    # write necessary fields into json dict and create json string
    if content or (content == "" and "content" not in json_dict):
        json_dict["content"] = content

    if creation_year is not None:
        json_dict["creation_year"] = creation_year

    if source_dataset or (source_dataset == "" and "source_dataset" not in json_dict):
        json_dict["source_dataset"] = source_dataset

    del source_dataset
    del content

    # initialize file
    creation_year = json_dict["creation_year"]
    if creation_year not in year_lengths:
        initialize_year(creation_year)

    # write entry
    path = year2path(creation_year)[1]
    with open(path, "a") as out_file:
        out_file.write("\n" if year_lengths[creation_year] == 0 else ",\n")
        year_lengths[creation_year] += 1
        out_file.write(json.dumps(json_dict))


undated_path = "./dataset/dataset_text_sequence/undated.json"
undated_count = 0


def report_undated_entry(json_dict={}):
    """
    call this function whenever there is an element whose year could not be automatically determined
    (either due to a failure to automatically parse the year number, or due to the lack of a date field)
    and requires later manual inspection. they will be saved to undated.json
    """
    global undated_count, undated_path

    if not os.path.isfile(undated_path):
        with open(undated_path, "w") as out_file:
            out_file.write("[")

    with open(undated_path, "a") as out_file:
        out_file.write("\n" if undated_count == 0 else ",\n")
        undated_count += 1
        out_file.write(json.dumps(json_dict))


def seal_all_files():
    """
    call this function after all writing of all text data files are finished.
    it is already called at the end of build_dataset.py, and you don't need to call it manually.
    """
    for yr_num in year_lengths:
        path = year2path(yr_num)[1]
        if year_lengths[yr_num] == 0:
            os.system(f"rm {path}")
        else:
            with open(path, "a") as out_file:
                out_file.write("\n]")

    if undated_count == 0:
        if os.path.exists(undated_path):
            os.system(f"rm {undated_path}")
    else:
        with open(undated_path, "a") as out_file:
            out_file.write("\n]")


def decode_year_num(date_str: str, year_lower_bound: int, year_upper_bound: int):
    """
    utility for robustly decoding year number from arbitrarily formatted dates; BC dates are supported (in which case a negative year number will be returned)
    please ALWAYS supply lower & upper bounds of year nubmers for validation purposes
    upon failure to extract a year number, None is returned
    please DO NOT use this function if the string is known to contain a year/date range, as opposed to one single year/date
    """
    try:
        pos = min(
            [date_str.index(s) for s in re.findall("[0-9][^0-9]", date_str)]
            + [len(date_str) - 1]
        )
        yr_str = date_str[: pos + 1]
        is_bc = "-" in yr_str or "bc" in date_str.lower().replace(".", "")
        creation_year = int("".join([ch for ch in yr_str if re.match("[0-9]", ch)])) * (
            -1 if is_bc else 1
        )
        assert year_lower_bound <= creation_year <= year_upper_bound
        return creation_year
    except:
        candidates = [
            int(s)
            for s in re.split("[^0-9]", date_str)
            if s and (year_lower_bound <= int(s) <= year_upper_bound)
        ]
        return candidates[0] if len(candidates) == 1 else None


def read_json_memory_efficient(path: str) -> Iterable[Dict[Hashable, Any]]:
    """
    Read a json file with the format List[Dict[Hashable, Any]] (e.g. List[Dict[str,List[str]]]).
    Instead of reading the entire file into memory, it reads chunk by chunk (with no regard to line breaks) and returns an iterator.
    It does not require that each Dict rests on a single line, nor that only one Dict rests on one line.
    """
    stack = []
    last_key = None

    def catch_all_events(event_name, *args):
        # print('\t{} : {}'.format(event_name, args))
        nonlocal stack, last_key
        if "doc" in event_name:
            return

        if "start" in event_name:
            obj = {} if event_name == "object_start" else []

            if len(stack) and last_key is not None:
                assert type(stack[-1]) == dict
                stack[-1][last_key] = obj
                last_key = None
            elif len(stack):
                assert type(stack[-1]) == list
                stack[-1].append(obj)

            stack.append(obj)
        elif "end" in event_name:
            if len(stack) != 1:
                assert type(stack[-1]) == (list if "array" in event_name else dict)
                stack.pop()
        elif event_name == "key":
            assert last_key is None and (type(stack[-1]) == dict)
            last_key = args[0]
        elif event_name == "value":
            assert last_key is not None and (type(stack[-1]) == dict)
            stack[-1][last_key] = args[0]
            last_key = None
        else:
            assert event_name == "element" and (type(stack[-1]) == list)
            stack[-1].append(args[0])

    streamer = JSONStreamer()
    streamer.add_catch_all_listener(catch_all_events)

    max_chars = 500000
    with open(path, "r") as in_file:
        while True:
            s = in_file.read(max_chars)
            if not s:
                break

            streamer.consume(s)
            assert len(stack) and (type(stack[0]) == list)
            if len(stack[0]):
                for i in range(len(stack[0]) - 1):
                    yield stack[0][i]

                stack[0] = [stack[0][-1]]

    streamer.close()
    for element in stack[0]:
        yield element


class JsonListWriter:

    def __init__(self, json_path: str):
        """Write a json list into a file, line by line. Memory-efficient, and can handle arbitrarily large files.
        This class should be used as a context manager.
        Example:
        with JsonListWriter('./test.json') as writer:
            writer.append({'key': value})"""

        self.path = json_path
        self.file_obj = open(self.path, "w")
        self.file_obj.write("[")
        self.is_first = True

    def __enter__(self):
        return self

    def append(self, element: Any):
        """Append an element at the end of the json list."""

        self.file_obj.write("\n" if self.is_first else ",\n")
        self.is_first = False
        self.file_obj.write(json.dumps(element))

    def __exit__(self, type, value, traceback):
        self.file_obj.write("\n]")
        self.file_obj.close()
