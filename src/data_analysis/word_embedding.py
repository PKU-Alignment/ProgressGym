import os
import json
import numpy as np
import scipy
from gensim.models import word2vec
import collections
import re


def readfile(path):
    # 读取文件夹下所有文件
    files = os.listdir(path)
    file_list = []
    for file in files:
        if not os.path.isdir(file):
            file_list.append(path + "/" + file)
    return file_list


"""
def clean_data(file_list):
    #读取文件夹内所有json文件，去除停用词，记录到words中
    
    linelist = []
    for file in file_list:
        with open(file,'r',encoding='utf-8') as fr:
            lines = json.load(fr)
            linelist.append(lines)
    words = []
    with open ('/mnt/models-pku/progressalign/xuchuan/stop_word_list_English.txt','r',encoding='utf-8') as f:
        stopWords = f.readlines()
    for lines in linelist:
        for line in lines:
            wordlist = line['content'].split()
            for word in wordlist:
                word = word.strip('\n').strip()
                if word in stopWords:
                    continue
                words.append(word)
    f.close()
    return words
    
    for file_path in file_list:
        with open(file_path,'r') as fr:
            lines = json.load(fr)
        text = [line['content'].strip().split(' ') for line in lines]
        fr.colse()
    """


def stat_words(file_list, freq_path):
    """
    统计词频保存到文件，了解数据集基本特征
    Args:
        words: 词汇list
        freq_path: 词频文件保存路径
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
    fw = open(freq_path, "w")  # 将词频数据保存到文件
    for i in range(len(word_freq_list)):
        content = " ".join(
            str(word_freq_list[i][j]) for j in range(len(word_freq_list[i]))
        )
        fw.write(content + "\n")
    fw.close()
    # print(type(word_freq_list[0]))--list
    return words, word_freq_list


def get_word_embedding(words, model_path):
    """
    利用gensim库生成语料库word embedding
    Args:
        input_corpus: 语料库文件路径
        model_path: 预训练word embedding文件保存路径
    """
    sentences = word2vec.Text8Corpus(words)  # 加载语料
    # vector_size词向量维度、window滑动窗口大小上下文最大距离、min_count最小词频数、epochs随机梯度下降迭代最小次数
    model = word2vec.Word2Vec(
        sentences, vector_size=100, window=8, min_count=3, epochs=8
    )
    model.save(model_path)
    model.wv.save_word2vec_format(model_path, binary=False)


if __name__ == "__main__":
    corpus_path = "/mnt/models-pku/progressalign/xuchuan/temp_1json"  # 语料文件夹路径
    file_list = readfile(corpus_path)
    freq_path = (
        "/mnt/models-pku/progressalign/xuchuan/words_freq_info.txt"  # 词频文件保存路径
    )
    words, word_freq_list = stat_words(
        file_list, freq_path
    )  # 统计保存预料中词频信息并保存
    model_path = "/mnt/models-pku/progressalign/xuchuan/word_embedding.bin"  # 训练词向量文件保存路径
    get_word_embedding(words, model_path)  # 训练得到预料的词向量
