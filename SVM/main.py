from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import pandas as pd
import sys
import jieba
import time
import os
import torch

stop_words = []
train = sys.argv[1]
test = sys.argv[2]
output = sys.argv[3]
dataset_name = sys.argv[4]


# train = '../data/two/train.csv'
# test = '../data/two/test.csv'
# output = '../output/two/SVM'
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


from sklearn.feature_extraction.text import TfidfVectorizer

# corpus = [
#     'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]

train_corpus = [' '.join(se) for se in train_sentences]
vectorizer = TfidfVectorizer()
vectorizer.fit(train_corpus)

train_sentences = vectorizer.transform(train_corpus).todense()
train_sentences = np.asarray(train_sentences)

test_corpus = [' '.join(se) for se in test_sentences]
test_sentences = vectorizer.transform(test_corpus).todense()
test_sentences = np.asarray(test_sentences)

labels = list(set(train_labels))
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(labels)
train_labels = label_encoder.transform(train_labels)
test_labels = label_encoder.transform(test_labels)


train_start_time = time.time()                             #定义模型开始训练时间

from sklearn.svm import LinearSVC
linearsvc = LinearSVC(C=1e9)
linearsvc.fit(train_sentences, train_labels)

train_end_time = time.time()                              #模型训练结束时间


predict_start_time = time.time()                          #预测开始时间

res01 = linearsvc.score(train_sentences, train_labels)                    #评估训练集的准确度
res02 = linearsvc.score(test_sentences, test_labels)                      #评估测试集的准确度
print('训练集合上acc:', res01)
print('测试集合上acc:', res02)
predict_end_time = time.time()                            #预测结束时间

print('The training: {}'.format(res01))
print('The test: {}'.format(res02))                       #输出结果

print('train time(s):', train_end_time-train_start_time)     #训练时间
print('test time(s):',predict_end_time-predict_start_time)   #预测时间


# Save it to gdrive
os.makedirs(output, exist_ok=True)
print(output)
save_dict = {
    'model': linearsvc,
    'label_encoder': label_encoder,
    'vectorizer': vectorizer,
    'stop_words': stop_words
}
torch.save(save_dict, f'{output}/SVM.pth')

predict_y = linearsvc.predict(test_sentences)

with open(f'{output}/SVM.txt', "w") as filer:
    filer.write(metrics.classification_report(test_labels, predict_y))

with open(f'{output}/acc.txt', "w") as filer:
    filer.write(str(metrics.accuracy_score(test_labels, predict_y)))