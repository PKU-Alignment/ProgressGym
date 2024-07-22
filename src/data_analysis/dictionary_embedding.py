from sentence_transformers import SentenceTransformer, util

# from word2vec_pca import readfile
import re
import json
import numpy as np
import pandas as pd
import os

# sentences = ["This is a sentence","each sentence is good"]

# model = SentenceTransformer('sentence-transformers/sentence-t5-base')
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
# embeddings = model.encode(sentences)
# print(len(embeddings))
# print(len(embeddings[0]))


def readfile(path):
    # 读取文件夹下所有文件
    files = os.listdir(path)
    file_list = []
    for file in files:
        if not os.path.isdir(file):
            file_list.append(path + "/" + file)
    return file_list


def sentence_embedding(lines, i):
    text = re.split("[.]|[,]|[;]|[?]|[!]|[。][\n]", lines[i]["content"].strip())
    # text = [line['content'].strip().split('[.?,;]') for line in lines]

    embeddings = model.encode(text, batch_size=1280)
    embeddings = np.asarray(embeddings)
    year_vec = np.mean(embeddings, axis=0)

    return year_vec


if __name__ == "__main__":
    corpus_path = "/mnt/models-pku/progressalign/shared_storage/our_datasets/HisText_Apr9_Guten_EEBO_PoL_IA10_Mistral7B_refined/C021/Y02021.json"  # 语料文件夹路径
    # file_list = readfile(corpus_path)
    # file_list.sort()
    # print(file_list)

    # for file in file_list:
    with open(corpus_path, "r") as fr:
        lines = json.load(fr)
    dict_vec_list = []
    index_list = []
    # print(len(lines))
    for i in range(len(lines)):
        dict_vec_list.append(sentence_embedding(lines, i))
        index_list.append(corpus_path[-11:-5] + "_" + str(i + 1))
        print(corpus_path[-11:-5] + "_" + str(i + 1))
    fr.close()
    print(corpus_path[-11:-5])
    df = pd.DataFrame(dict_vec_list, index=index_list)
    df.to_csv(
        "/mnt/models-pku/progressalign/xuchuan/ProgressAlign/text embedding/dict_vec_list/"
        + corpus_path[-11:-5]
        + ".csv",
        sep=",",
        index=True,
        header=False,
    )
