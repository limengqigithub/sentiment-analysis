import os
import sys
import pandas as pd
import torch
import jieba
import numpy as np
test = sys.argv[1]
# with open(test, 'r', encoding='utf-8') as f:
#     first_line = f.readline()
# if ';' in first_line:
#     test_df = pd.read_csv(test, delimiter=";")
# elif ',' in first_line:
#     test_df = pd.read_csv(test, delimiter=",")
test_df = pd.read_csv(test, delimiter=";")
# Load pretrained model for specific dataset
model = torch.load(sys.argv[2]).get('model')
label_encoder = torch.load(sys.argv[2]).get('label_encoder')
dictionary = torch.load(sys.argv[2]).get('dictionary')
stop_words = torch.load(sys.argv[2]).get('stop_words')
# save_dict = {
#     'model': forest,
#     'label_encoder': label_encoder,
#     'dictionary': dictionary,
#     'stop_words': stop_words
# }

sentences = test_df.Text.values
# labels = test_df.Polarity.values
# ids = test_df.id.values
# 去处停用词，并返回评论切分后的评论列表，每个元素均为一个词语列表
def removeStopWords(X, stop_words):
    all_words = []  # 定义存放评论的列表，评论均以词典列表形式存在
    for sentence in X:  # 遍历评论列表
        words = []  # 存放每个评论词语的列表
        for word in jieba.lcut(sentence):  # 遍历评论分词后的列表
            if word not in stop_words:  # 该词语未在停用词表中
                words.append(word)  # 追加到words中，实现去停用词
        all_words.append(words)  # 总评论列表追加该评论分词列表
    return all_words  # 返回结果
print(stop_words)
print(sentences)
sentences = removeStopWords(sentences, stop_words)


# 通过词袋模型将每条评论均转换为对应的向量，返回
def word_2_vec(X, dictionary):
    n = len(dictionary)  # 词典长度

    word_vecs = []  # 存放所有向量的列表

    for sentence in X:  # 遍历评论列表
        word_vec = np.zeros(n)  # 生成 (1,n)维的 0向量
        for word in sentence:  # 遍历评论中的所有词语
            if word in dictionary:  # 若该词语在字典中
                loc = dictionary.index(word)  # 找到在词典中的位置
                word_vec[loc] += 1  # 为向量的此位置累加1

        word_vecs.append(word_vec)  # 评论列表累加向量

    return np.array(word_vecs)  # 返回结果

train_X = word_2_vec(sentences, dictionary)

predict_y = model.predict(train_X)
print('预测样本数量:', len(sentences))

with open(sys.argv[3], "w") as filer:
    filer.write(','.join(label_encoder.inverse_transform(predict_y)))