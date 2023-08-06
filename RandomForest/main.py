from sklearn.ensemble import RandomForestClassifier  # 导入sklearn包下的随机森林分类器
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



# 生成词典，存放评论中所有出现过的词语，待统计词频
def getDictionary(X):
    dictionary = []  # 定义词典，存放所有出现过的词语

    for sentence in X:
        for word in sentence:
            if word not in dictionary:  # 遍历所有评论的词语，若未在词典中存在
                dictionary.append(word)  # 添加

    return dictionary  # 返回结果
dictionary = getDictionary(train_sentences)


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
train_X = word_2_vec(train_sentences, dictionary)
test_X = word_2_vec(test_sentences, dictionary)

labels = list(set(train_labels))
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(labels)
train_labels = label_encoder.transform(train_labels)
test_labels = label_encoder.transform(test_labels)


train_start_time = time.time()                             #定义模型开始训练时间

forest = RandomForestClassifier(n_estimators=100)                     #构造随机森林，共存在100棵树
forest.fit(train_X, train_labels)                                            #训练数据

train_end_time = time.time()                              #模型训练结束时间


predict_start_time = time.time()                          #预测开始时间

res01 = forest.score(train_X, train_labels)                    #评估训练集的准确度
res02 = forest.score(test_X, test_labels)                      #评估测试集的准确度
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
    'model': forest,
    'label_encoder': label_encoder,
    'dictionary': dictionary,
    'stop_words': stop_words
}
torch.save(save_dict, f'{output}/RandomForest.pth')

predict_y = forest.predict(test_X)

with open(f'{output}/RandomForest.txt', "w") as filer:
    filer.write(metrics.classification_report(test_labels, predict_y))

with open(f'{output}/acc.txt', "w") as filer:
    filer.write(str(metrics.accuracy_score(test_labels, predict_y)))