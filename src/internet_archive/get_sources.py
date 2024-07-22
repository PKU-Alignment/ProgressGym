import os, json, requests
import src.text_writer as tw
from tqdm import tqdm
import re
import time


def build_internet_archive_LibOfCong(max_hours: int = None):
    ia_path = "./src/internet_archive/ia"
    identifier_list_path = (
        "./dataset/raw_downloads/internet_archive_identifier_list_libofcong.txt"
    )
    os.system(f"chmod +x {ia_path}")

    # create list of identifiers from all Library of Congress books
    if not os.path.exists(identifier_list_path):
        print(f"downloading index file...")
        os.system(
            f'$({ia_path} search "collection:library_of_congress" --field=".txt" --itemlist > {identifier_list_path})'
        )
        print(f"download complete.")
    else:
        print(f"index file already downloaded; continue.")

    # initialize baseline URL, master list, master json
    base_url = "http://archive.org/metadata/"
    # list_master = []
    # json_master = {"master_list": list_master}

    # create list of identifiers from identifier_list.txt
    with open(identifier_list_path, "r") as f:
        identifiers = f.read().splitlines()

    example_counter = 0
    nodate_counter = 0
    download_fail_counter = 0
    start_time = time.monotonic()

    for identifier in tqdm(identifiers):
        url = f"{base_url}{identifier}"
        example_counter += 1

        if max_hours is not None:
            if time.monotonic() - start_time > 3600 * max_hours:
                print("time is up! ending loop.")
                tw.write_log("IA-LOC: time is up! ending loop.")
                break

            if example_counter % 100 == 2:
                tw.write_log(
                    "IA-LOC: time %.0f/%.0f = %.2f%%, progress %d/%d = %.2f%%"
                    % (
                        time.monotonic() - start_time,
                        3600 * max_hours,
                        (time.monotonic() - start_time) / (3600 * max_hours) * 100,
                        example_counter,
                        len(identifier),
                        example_counter / len(identifier) * 100,
                    )
                )

        try:
            if example_counter <= 200:
                tw.write_log(f"IA-LOC: {identifier} request #1 starts")
            response = requests.get(url)
            response.raise_for_status()
            if example_counter <= 200:
                tw.write_log(f"IA-LOC: {identifier} request #1 ends")

            content = response.json().get("files", [])
            content_strings = []

            for file_info in content:
                if file_info["name"].endswith(".txt"):
                    if example_counter <= 200:
                        tw.write_log(
                            f'IA-LOC: {identifier}-{file_info["name"]} request #2 starts'
                        )
                    file_url = (
                        f"http://archive.org/download/{identifier}/{file_info['name']}"
                    )
                    file_response = requests.get(file_url)
                    file_response.raise_for_status()
                    if example_counter <= 200:
                        tw.write_log(
                            f'IA-LOC: {identifier}-{file_info["name"]} request #2 ends'
                        )
                    file_content = file_response.text
                    content_strings.append(file_content)

            metadata = response.json().get("metadata", {})

            if "date" not in metadata:
                nodate_counter += 1
                tw.write_log(
                    f'IA-LOC: {nodate_counter}-th time, metadata contains no "date" field: {metadata}'
                )
                continue

            date_str: str = metadata["date"]
            creation_year = tw.decode_year_num(date_str, 510, 2024)

            if example_counter <= 200 or (
                type(creation_year) == int and creation_year < 500
            ):
                tw.write_log(
                    f"IA-LOC: date {date_str} interpreted as year {creation_year}"
                )

            metadata["creation_year"] = creation_year
            metadata["content"] = "\n\n===============\n\n".join(content_strings)
            metadata["source_dataset"] = "Internet_Archive"
            metadata["source_dataset_detailed"] = "Internet_Archive_LibOfCong"

            if creation_year is None:
                nodate_counter += 1
                tw.write_log(
                    f"IA-LOC: {nodate_counter}-th time, date {date_str} uninterpretable; saving to undated.json"
                )
                tw.report_undated_entry(json_dict=metadata)
            else:
                tw.write_single_entry(json_dict=metadata)

        # raise an exception if request fails
        # except requests.exceptions.RequestException as e:
        except Exception as e:
            download_fail_counter += 1
            tw.write_log(
                f"IA-LOC: {download_fail_counter}-th time, error fetching metadata for {identifier}: {type(e)} {e}"
            )

    # write json to a file
    # with open("final_json.json", "w", encoding="utf-8") as json_file:
    #     json.dump(json_master, json_file, indent=4)
