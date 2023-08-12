import jieba
import joblib
import os
import warnings
import pandas as pd
from sklearn import metrics
import sys
import torch
# 忽略UserWarning
warnings.filterwarnings("ignore")



stop_words = []
train = sys.argv[1]
test = sys.argv[2]
output = sys.argv[3]
dataset_name = sys.argv[4]


# train = '../data/two/train.csv'
# test = '../data/two/test.csv'
# output = '../output/two/NaiveBayes'
# dataset_name = 'two'

with open(train, 'r', encoding='utf-8') as f:
    first_line = f.readline()
if ';' in first_line:
    train_df = pd.read_csv(train, delimiter=";")
    test_df = pd.read_csv(test, delimiter=";")
elif ',' in first_line:
    train_df = pd.read_csv(train, delimiter=",")
    test_df = pd.read_csv(test, delimiter=",")


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


train_sentences = train_df.Text.values
train_labels = train_df.Polarity.values
train_sentences = removeStopWords(train_sentences, stop_words)
print('训练样本数量:', len(train_sentences))


test_sentences = test_df.Text.values
test_labels = test_df.Polarity.values
test_sentences = removeStopWords(test_sentences, stop_words)
print('测试样本数量:', len(test_sentences))


train_sentences = [' '.join(ts) for ts in train_sentences]
test_sentences = [' '.join(ts) for ts in test_sentences]

from sklearn.feature_extraction.text import CountVectorizer


# 特征提取
vect = CountVectorizer(max_df=0.8, min_df=3)
# 划分数据集


# 训练模型
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
x_train_vect = vect.fit_transform(train_sentences)
nb.fit(x_train_vect, train_labels)
train_score = nb.score(x_train_vect, train_labels)
print(train_score)

# 测试模型
x_test_vect = vect.transform(test_sentences)
test_score = nb.score(x_test_vect, test_labels)
# print(test_score)


os.makedirs(output, exist_ok=True)
save_dict = {
    'model': nb,
    'vect':vect,
    'stop_words': stop_words
}
torch.save(save_dict, f'{output}/NaiveBayes.pth')

predict_y = nb.predict(x_test_vect)
label_names = list(set(train_labels))
import numpy as np
def covrt(data):
    zeros = np.zeros(shape=(len(test_labels), len(label_names)))
    for i, d in enumerate(data):
        zeros[i][label_names.index(d)] = 1
    return zeros
with open(f'{output}/NaiveBayes.txt', "w") as filer:
    filer.write(metrics.classification_report(covrt(test_labels), covrt(predict_y), target_names=label_names))

with open(f'{output}/acc.txt', "w") as filer:
    filer.write(str(metrics.accuracy_score(test_labels, predict_y)))