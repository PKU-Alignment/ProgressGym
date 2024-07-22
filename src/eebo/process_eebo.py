import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import json
import src.text_writer as tw


# a utility function called by build_eebo_dataset, to read the contents in an eebo xml file in a suitable manner
def process_eebo_file(path: str):
    tree = ET.parse(path)
    root = tree.getroot()

    def remove_tag(root: ET.Element, tagname: str):
        for tag in list(root.iter(tagname)):
            assert tag.find(tagname) is None
            tag.clear()

    remove_tag(root, "HEADER")
    remove_tag(root, "IDG")
    remove_tag(root, "HEAD")
    remove_tag(root, "GAP")

    out_str = ET.tostring(root, encoding="utf8", method="text")
    return out_str.decode("utf8")


# if download_eebo is already called, build_eebo_dataset is the only thing you need to call to build the EEBO dataset
def build_eebo_dataset(eebo_path: str = "./dataset/raw_downloads/EEBO/"):
    for phase_num in [1, 2]:
        print(f"start building dataset from EEBO Phase {phase_num} (2 in total)")
        root_dir = os.path.join(eebo_path, f"eebo_phase{phase_num}")

        if phase_num == 1:
            index_file = os.path.join(root_dir, "eebo_phase1_IDs_and_dates.txt")
            search_folder = os.path.join(root_dir, "P4_XML_TCP")
        else:
            index_file = os.path.join(root_dir, "EEBO_Phase2_IDs_and_dates.txt")
            search_folder = os.path.join(root_dir, "P4_XML_TCP_Ph2")

        assert os.path.exists(index_file)
        assert os.path.exists(search_folder)

        file_list = [
            os.path.join(dirpath, filename)
            for dirpath, dirnames, filenames in os.walk(search_folder)
            for filename in filenames
            if "xml" in filename
        ]

        with open(index_file, "r") as in_file:
            index_lines = in_file.readlines()

        for line in tqdm(index_lines):
            if not line.strip():
                continue

            parts = line.strip().split("\t")
            assert len(parts) == 2

            expected_filename, yearrange = tuple(parts)
            matched_files = [path for path in file_list if expected_filename in path]
            assert len(matched_files) <= 1

            if not matched_files:
                print(f"missing: {expected_filename}")

            # interpret the ambiguous year range specified in the original dataset, providing a best-guess year number, an upper bound, and a lower bound
            yearrange = yearrange.replace("?", "")
            if "-" in yearrange:
                yearrange_parts = yearrange.split("-")
                assert len(yearrange_parts) == 2
                yearrange_parts[0] = yearrange_parts[0].replace("u", "0")
                yearrange_parts[1] = yearrange_parts[1].replace("u", "9")
                year_earliest, year_latest = int(yearrange_parts[0]), int(
                    yearrange_parts[1]
                )
                year = (year_earliest + year_latest + 1) // 2
            elif "u" in yearrange:
                earliest = yearrange.replace("u", "0")
                latest = yearrange.replace("u", "9")
                year_earliest, year_latest = int(earliest), int(latest)
                year = (year_earliest + year_latest + 1) // 2
            else:
                year_earliest, year, year_latest = (
                    int(yearrange),
                    int(yearrange),
                    int(yearrange),
                )

            if year_earliest > year_latest:
                year_earliest, year_latest = year_latest, year_earliest

            content = process_eebo_file(matched_files[0])

            json_element = {
                "content": content,
                "creation_year": year,
                "creation_year_earliest": year_earliest,
                "creation_year_latest": year_latest,
                "source_dataset": "EEBO",
                "source_dataset_detailed": f"EEBO_Phase{phase_num}",
            }

            if (
                1000 < year_earliest <= year <= year_latest < 2025
                and year_latest - year_earliest < 50
            ):
                tw.write_single_entry(json_dict=json_element)
            else:
                del json_element["creation_year"]
                tw.write_log(
                    f"EEBO: Uncertainty too large, saving to undated.json: {line.strip()}"
                )
                tw.report_undated_entry(json_dict=json_element)
