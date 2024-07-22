import pandas as pd
import matplotlib.pyplot as plt
import struct
import os
import json
import numpy as np
import gensim
import collections
from gensim.models import word2vec
from gensim.models import KeyedVectors

# import the PCA module from sklearn
from sklearn.decomposition import PCA

import gensim.downloader as api

wv = api.load("word2vec-google-news-300")
# google-news-300的embedding


def readfile(path):
    # 读取文件夹下所有文件
    files = os.listdir(path)
    file_list = []
    for file in files:
        if not os.path.isdir(file):
            file_list.append(path + "/" + file)
    return file_list


def stat_words(file_list):
    """
    统计词频保存到文件，了解数据集基本特征
    Return:
        word_list = [[word:count],...]
    """
    with open(
        "/mnt/models-pku/progressalign/xuchuan/stop_word_list_English.txt", "r"
    ) as f:
        stopWords = [line.strip() for line in f.readlines()]
    word_count = collections.Counter()
    words = []
    for file_path in file_list:
        with open(file_path, "r") as fr:
            lines = json.load(fr)
        text = [line["content"].strip().split(" ") for line in lines]
        print(type(text))
        fr.close()
        for content in text:
            # print(type(content))
            for word in content:
                # print(word)
                # print(type(content))  原本是list，上面强转成str
                if word == "" or word == "\n":
                    continue
                # print(content)
                wordtmp = ""
                for s in range(len(word)):
                    if (word[s] >= "a" and word[s] <= "z") or (
                        word[s] >= "A" and word[s] <= "Z"
                    ):
                        wordtmp += word[s]
                if wordtmp in stopWords:
                    continue
                # print(type(wordtmp))
                words.append(wordtmp)
    word_count.update(words)
    word_freq_list = sorted(word_count.most_common(), key=lambda x: x[1], reverse=True)
    # fw = open(freq_path, 'w') #将词频数据保存到文件
    # for i in range(len(word_freq_list)):
    #     content = ' '.join(str(word_freq_list[i][j]) for j in range(len(word_freq_list[i])))
    #     fw.write(content + '\n')
    # fw.close()
    # print(type(word_freq_list[0]))--list
    return word_freq_list


def cal_10year_vec(words_freq_list):
    years_vec = np.zeros((1, 300))
    word_cnt = 0
    for i in range(len(word_freq_list)):
        try:
            years_vec += word_freq_list[i][1] * (
                wv[word_freq_list[i][0]].reshape((1, 300))
            )
            word_cnt += word_freq_list[i][1]
        except:
            pass
    years_vec /= word_cnt
    # print(years_vec)
    # print(type(years_vec))
    return years_vec.tolist()[0]


def pca(year_vec_list):
    # with open(words_df_path,'r') as f:
    #   lines=f.readlines()
    # df = [line.split() for line in lines]
    # initialize the pca model
    pca = PCA(n_components=2)
    # years_df = pd.DataFrame(year_vec_list)
    year_vecs_2D = pca.fit_transform(year_vec_list)
    # tell our fitted pca model to transform our data dowm to 10D using the instructions it learnt during the fit phase
    # year_vecs_2D=pca.transform(years_df)
    # print(year_vecs_2D)
    return year_vecs_2D


if __name__ == "__main__":
    corpus_path = "/mnt/models-pku/progressalign/shared_storage/our_datasets/histext_19AD_collection_G"  # 语料文件夹路径
    file_list = readfile(corpus_path)
    year_vec_list = []
    for i in range(10):
        file_list_tmp = []
        for j in range(10):
            file_list_tmp.append(file_list[10 * i + j])
        word_freq_list = stat_words(file_list_tmp)  # 统计保存预料中词频信息
        year_vec_list.append(cal_10year_vec(word_freq_list))
    year_vec_2D = pca(year_vec_list)
