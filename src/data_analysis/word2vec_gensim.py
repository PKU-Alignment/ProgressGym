import numpy as np
import scipy
import json
import gensim
import collections
from gensim.models import word2vec
from gensim.models import KeyedVectors


def stat_words(file_path, freq_path):
    """
    统计词频保存到文件，了解数据集基本特征
    Args:
        file_path: 语料库文件路径
        freq_path: 词频文件保存路径
    Return:
        word_list = [[word:count],...]
    """
    with open(file_path, "r") as fr:
        lines = json.load(fr)
    # 从语料库文件中读取数据并统计词频
    text = [line["content"].strip().split(" ") for line in lines]
    fr.close()
    word_counts = collections.Counter()
    for content in text:
        word_counts.update(content)
    word_freq_list = sorted(word_counts.most_common(), key=lambda x: x[1], reverse=True)
    fw = open(freq_path, "w")  # 将词频数据保存到文件
    for i in range(len(word_freq_list)):
        content = " ".join(
            str(word_freq_list[i][j]) for j in range(len(word_freq_list[i]))
        )
        fw.write(content + "\n")
    fw.close()
    return word_freq_list


def get_word_embedding(input_corpus, model_path):
    """
    利用gensim库生成语料库word embedding
    Args:
        input_corpus: 语料库文件路径
        model_path: 预训练word embedding文件保存路径
    """
    sentences = word2vec.Text8Corpus(input_corpus)  # 加载语料
    # vector_size词向量维度、window滑动窗口大小上下文最大距离、min_count最小词频数、epochs随机梯度下降迭代最小次数
    model = word2vec.Word2Vec(
        sentences, vector_size=100, window=8, min_count=3, epochs=8
    )
    model.save(model_path)
    model.wv.save_word2vec_format(model_path, binary=False)


if __name__ == "__main__":
    corpus_path = "../shared_storage/our_datasets/histext_1826_to_2018_collection_G/C019/Y01826.json"  # 中文预料文件路径
    freq_path = "words_freq_info.txt"  # 词频文件保存路径
    word_list = stat_words(corpus_path, freq_path)  # 统计保存预料中词频信息并保存
    model_path = "word_embedding.bin"  # 训练词向量文件保存路径
    get_word_embedding(corpus_path, model_path)  # 训练得到预料的词向量
