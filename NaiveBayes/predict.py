import sys
import torch
import jieba
import pandas as pd

test = sys.argv[1]


model = torch.load(sys.argv[2]).get('model')
vect = torch.load(sys.argv[2]).get('vect')
stop_words = torch.load(sys.argv[2]).get('stop_words')

test_df = pd.read_csv(test, delimiter=";")

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

train_sentences = removeStopWords(train_sentences, stop_words)


train_sentences = [' '.join(ts) for ts in train_sentences]


x_test_vect = vect.transform(train_sentences)

predict_y = model.predict(x_test_vect)


print('预测样本数量:', len(train_sentences))

with open(sys.argv[3], "w") as filer:
    filer.write(','.join(predict_y))