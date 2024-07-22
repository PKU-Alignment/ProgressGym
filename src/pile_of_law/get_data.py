import requests
from bs4 import BeautifulSoup
import os, json
import lzma
import src.text_writer as tw
from tqdm import tqdm
from copy import copy

"""
This file downloads compressed files from the dataset's official huggingface repo,
then extracts them and combine them into one giant json under the root folder.
"""

PoL_failure_counter = 0


def get_pile_of_law():
    """
    DOWNLOADING COMPRESSED FILES FROM HUGGINGFACE
    """
    compressed_dir = "./dataset/raw_downloads/pile_of_law_compressed"
    decompressed_dir = "./dataset/raw_downloads/pile_of_law_decompressed"
    if not os.path.isdir(compressed_dir):
        os.mkdir(compressed_dir)
    if not os.path.isdir(decompressed_dir):
        os.mkdir(decompressed_dir)

    def download_file(url, folder_path):
        # 从URL中获取文件名
        file_name = url.split("/")[-1].split("?")[0]
        file_path = os.path.join(folder_path, file_name)

        # 发起HTTP请求并下载文件
        tw.write_log(f"PileOfLaw: start downloading {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        tw.write_log(f"PileOfLaw: File '{file_name}' downloaded to '{folder_path}'")

    # 示例网页链接
    url = "https://huggingface.co/datasets/pile-of-law/pile-of-law/tree/main/data"
    # 发起HTTP请求获取网页内容
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # 提取所有文件下载链接
    file_links = [
        a["href"]
        for a in soup.find_all("a", href=True)
        if a["href"].endswith("download=true")
    ]  # 假设我们要下载PDF文件

    # 定义保存文件的文件夹路径
    download_folder = compressed_dir

    # 如果文件夹不存在，则创建它
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # 下载所有文件
    print("start downloading files.")
    for file_link in tqdm(file_links):
        this_url = "https://huggingface.co" + file_link
        file_name = this_url.split("/")[-1].split("?")[0]
        if os.path.isfile(os.path.join(download_folder, file_name)):
            tw.write_log(
                f"PileOfLaw: file {file_name} already downloaded. skipping to the next file in line."
            )
        else:
            download_file(this_url, download_folder)

    """
    DECOMPRESSING AND RENDERING DATA INTO ONE JSON
    """
    print("start decompressing")
    compressed = os.listdir(compressed_dir)
    for pack in tqdm(compressed):
        if not pack.endswith(".xz"):
            continue
        name = pack.split(".")[1]
        if os.path.exists(os.path.join(decompressed_dir, f"{name}.jsonl")):
            continue
        with lzma.open(os.path.join(compressed_dir, pack), "rb") as f_1, open(
            os.path.join(decompressed_dir, f"{name}.jsonl"), "wb"
        ) as f_2:
            for chunk in f_1:
                f_2.write(chunk)

    def read_jsonl_and_concat(jsonl_dir):
        global PoL_failure_counter
        jsonls = os.listdir(jsonl_dir)
        for id, jsonl in enumerate(jsonls):
            name = jsonl.split(".")[0]
            print(f"start processing {name}, {id+1}/{len(jsonls)}")

            # use iterator instead of reading entire file into memory
            with open(os.path.join(jsonl_dir, jsonl), "r") as f:
                for i, line in tqdm(enumerate(f)):

                    try:
                        dd = json.loads(line)

                        dd["content"] = dd.pop("text")
                        dd["culture"] = "English"
                        dd["source_dataset"] = "Pile_of_Law"

                        with open("./src/pile_of_law/source.json", "r") as f:
                            source_dict = json.load(f)
                            dd["source_dataset_detailed"] = "Pile_of_Law_" + name
                            dd["source_dataset_detailed_explanation"] = (
                                source_dict[name] if name in source_dict else ""
                            )

                        creation_year = None
                        if "created_timestamp" in dd and dd["created_timestamp"]:
                            # print(dd['created_timestamp'])
                            if (
                                type(dd["created_timestamp"]) == int
                                and 1000 <= dd["created_timestamp"] <= 2024
                            ):
                                creation_year = dd["created_timestamp"]

                            if type(dd["created_timestamp"]) == str:
                                creation_year = tw.decode_year_num(
                                    dd["created_timestamp"], 1100, 2024
                                )

                        if creation_year is None:
                            tw.report_undated_entry(dd)
                            PoL_failure_counter += 1
                            tw.write_log(
                                f'PileOfLaw: {PoL_failure_counter}-th time, saving to undated.json: created_timestamp={dd["created_timestamp"] if "created_timestamp" in dd else None}'
                            )
                        else:
                            dd["creation_year"] = creation_year
                            tw.write_single_entry(json_dict=dd)

                    except Exception as e:
                        PoL_failure_counter += 1
                        tw.write_log(
                            f"PileOfLaw: {PoL_failure_counter}-th time, error processing metadata for {i}-th entry of {name}: {type(e)} {e}"
                        )

    read_jsonl_and_concat(decompressed_dir)
