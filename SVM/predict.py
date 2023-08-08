import numpy as np
import pandas as pd
import sys
import jieba
import torch

stop_words = []
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
vectorizer = torch.load(sys.argv[2]).get('vectorizer')
stop_words = torch.load(sys.argv[2]).get('stop_words')


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

train_sentences = test_df.Text.values
sentences = removeStopWords(train_sentences, stop_words)



from sklearn.feature_extraction.text import TfidfVectorizer

# corpus = [
#     'This is the first document.',
#     'This document is the second document.',
#     'And this is the third one.',
#     'Is this the first document?',
# ]

train_corpus = [' '.join(se) for se in sentences]

train_sentences = vectorizer.transform(train_corpus).todense()
train_X = np.asarray(train_sentences)
predict_y = model.predict(train_X)
print('预测样本数量:', len(sentences))

with open(sys.argv[3], "w") as filer:
    filer.write(','.join(label_encoder.inverse_transform(predict_y)))