import src.text_writer as tw
import os, json
import csv
from tqdm import tqdm

"""
This file is to be excecuted after get_data.py. Reads metadata from raw files and writes them via text_writer API
"""


def gather_meta(raw_dir, record):
    gutenberg_failure_counter = 0
    # all_add = []
    gathered = []
    """
    if os.path.exists(record):
        with open(record, 'r') as f:
            gathered = f.readlines()
    """
    catalog = os.listdir(raw_dir)
    print(
        f"Gutenberg: finalize dataset, len(Catalog)={len(catalog)}, len(Gathered)={len(gathered)}"
    )
    for ele in tqdm(catalog):
        try:
            if ele in gathered:
                continue
            with open(record, "a+") as f:
                f.write(ele + "\n")
            add = {
                "source_document": "",
                "creation_year": "",
                "culture": "",
                "content": "",
                "source_dataset": "gutenberg",
            }
            full_timestamp = "[NA]"
            with open(os.path.join(raw_dir, ele), "r") as f:
                text = f.readlines()
                for i, line in enumerate(text):
                    if line.startswith("Title"):
                        add["source_dataset_detailed"] = (
                            "gutenberg - " + line.split(":")[1]
                        )
                    # elif line.startswith("Release date"):
                    #    add["created_timestamp"] = line.split(':')[1].split('[')[0]
                    #    full_timestamp = line.split(':')[1]
                    elif line.startswith("Language"):
                        try:
                            add["culture"] = line.split(":")[1]
                        except:
                            add["culture"] = None
                    elif line.startswith("*** START"):
                        add["content"] = "".join(text[i + 1 :])
                        break

            assert add["content"]
            # add['creation_year'] = tw.decode_year_num(add["created_timestamp"], 1100, 2024)
            """
            Taking average from the author's y.o.b & y.o.d
            """
            with open(
                os.path.join(
                    "dataset", "raw_downloads", "Gutenberg", "metadata", "metadata.csv"
                )
            ) as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[0] == ele.split("_")[0]:
                        try:
                            add["creation_year"] = str(
                                (int(row[3]) + int(20) + int(row[4])) // 2
                            )
                        except:
                            if row[3] != "":
                                add["creation_year"] = int(row[3]) + 30
                            elif row[4] != "":
                                add["creation_year"] = int(row[4]) - 20
                            else:
                                add["creation_year"] = None
                        break
            if add["creation_year"] is not None:
                tw.write_single_entry(json_dict=add)
            else:
                tw.report_undated_entry(add)
                gutenberg_failure_counter += 1
                if (
                    gutenberg_failure_counter <= 100
                    or gutenberg_failure_counter % 100 == 0
                ):
                    tw.write_log(
                        f'Gutenberg: {gutenberg_failure_counter}-th time, saving to undated.json: created_timestamp={add["created_timestamp"]},{full_timestamp}'
                    )

        except Exception as e:
            gutenberg_failure_counter += 1
            tw.write_log(
                f"Gutenberg: {gutenberg_failure_counter}-th time, exception {type(e)} {e}"
            )
