import os, json
import re, unicodedata
import src.text_writer as tw
from tqdm import tqdm

last_cleaned = ""


def sentence_determinant(sentence):
    # 使用正则表达式匹配非文字字符
    count = sum(
        1 for char in sentence if not unicodedata.category(char).startswith("L")
    )
    # 统计匹配到的非文字字符数量
    if len(sentence) == 0:
        return 0
    if count / len(sentence) > 0.6:
        return 0
    else:
        return 1


def json_generator(file):
    buffer = ""
    for line in file:
        buffer += line.strip()
        try:
            # 尝试解析缓冲区中的内容为一个JSON对象
            obj = json.loads(buffer)
            yield obj
            # 重置缓冲区
            buffer = ""
        except json.JSONDecodeError:
            # 如果无法解析为JSON对象，则继续读取下一行
            continue


def cleanse_text(text):
    """
    text in the form of a giant str
    """
    text = text.strip()
    sentences = text.split("\n")
    out_sentences = []
    global last_cleaned
    for sentence in sentences:
        if "*** END OF THE PROJECT GUTENBERG EBOOK" in sentence:
            last_cleaned = sentence
            break
        if "http://" in sentence or "INDEX" in sentence or "Index" in sentence:
            # print("sentence to be killed ", sentence)
            last_cleaned = sentence
            continue
        if not sentence_determinant(sentence):
            # print("sentence to be killed ", sentence)
            last_cleaned = sentence
            continue
        out_sentences.append(sentence)
    return "\n".join(out_sentences)


def cleanse_dir(dirct, to_dir):
    os.makedirs(to_dir)
    for year in tqdm(os.listdir(dirct), desc=dirct.split("/")[-1]):
        generator = tw.read_json_memory_efficient(os.path.join(dirct, year))
        out = []
        for boi in generator:
            orig_len = len(boi["content"])
            boi["content"] = cleanse_text(boi["content"])

            tw.write_log(
                "cleansed an object in "
                + year
                + ", length reduced from "
                + str(orig_len)
                + " to "
                + str(len(boi["content"]))
                + "; last cleaned "
                + last_cleaned[:250]
            )

            if len(boi["content"]) > 200:
                out.append(boi)
            else:
                tw.write_log(f"Ignoring {repr(boi['content'])}.")

        with open(os.path.join(to_dir, year), "w") as file:
            json.dump(out, file)


def cleanse(dataset_path, to_path):
    for century in os.listdir(dataset_path):
        if not os.path.isdir(os.path.join(dataset_path, century)):
            continue
        cleanse_dir(os.path.join(dataset_path, century), os.path.join(to_path, century))


if __name__ == "__main__":
    cleanse(
        "../../shared_storage/our_datasets/HisText_Mar8_Guten_EEBO_PoL_IA10_unrefined/",
        "../../shared_storage/our_datasets/HisText_Mar8_Guten_EEBO_PoL_IA10_rulebased_refined/",
    )
