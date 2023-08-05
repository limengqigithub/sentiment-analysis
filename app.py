import os
import io
import pandas as pd
from flask import Flask, request, send_from_directory

app = Flask(__name__)


# 上传数据集接口
@app.route('/file_upload', methods=['POST'])
def file_upload():
    files = request.files.getlist('file')
    dataset_name = request.form.get('dataset_name')
    if files and dataset_name:
        for file in files:
            save_path = os.path.join('data', dataset_name)
            os.makedirs(save_path, exist_ok=True)
            file.save(os.path.join(save_path, file.filename))
        return '文件上传成功'
    else:
        return '上传失败，未选择文件或未命名数据集名称'


# 提交训练任务接口
@app.route('/train', methods=['POST'])
def train():
    dataset_name = request.form.get('dataset_name')
    model_lists = request.form.get('model_lists')
    name_2_acc = {}
    if not dataset_name and not model_lists:
        return '参数提交缺失'
    else:
        for model in model_lists.split(';'):
            if model == 'RoBERTa':
                print('开始训练: {}.'.format(model))
                os.system('python ./RoBERTa/main.py ./data/{}/train.csv ./data/{}/test.csv ./output/{}/{} {}'.format(dataset_name, dataset_name, dataset_name, model, dataset_name))
                print('{}:训练完成'.format(model))
                name_2_acc[model] = open('./output/{}/{}/acc.txt'.format(dataset_name, model),'r').read()
            if model == 'SVM':
                print('SVM 模型未实现。')
        return name_2_acc


@app.route("/file_download", methods=["POST", "GET"])
def download_file():
    dataset_name = request.form.get('dataset_name')
    model = request.form.get('model')
    type = request.form.get('type')
    return send_from_directory(f'./output/{dataset_name}/{model}', model + '.' + type, as_attachment=True)


# 预测的接口
@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('file')
    dataset_name = request.form.get('dataset_name')
    model_lists = request.form.get('model_lists')
    for file in files:
        file.save('./demo/temp.csv')
        # df = pd.read_csv(file.stream, delimiter=';')
        # print(df)
    name_2_label = {}
    if not dataset_name and not model_lists and not files:
        return '参数提交缺失'
    else:
        for model in model_lists.split(';'):
            if model == 'RoBERTa':
                print('开始使用推理: {}.'.format(model))
                os.system('python ./RoBERTa/predict.py ./demo/temp.csv ./output/{}/{}/{}.pth ./demo/result.txt'.format(dataset_name, model, model))
                print('{}:推理完成'.format(model))
                with open('demo/result.txt', 'r', encoding='utf-8') as f:
                    name_2_label[model] = f.read().strip().split(',')
            if model == 'SVM':
                print('SVM 模型未实现。')
        return name_2_label




if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8868", debug=False)