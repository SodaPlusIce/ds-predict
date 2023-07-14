from django.http import HttpResponse
import csv
from django.http import JsonResponse
import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from imblearn.over_sampling import RandomOverSampler, KMeansSMOTE
from django.http import FileResponse
import os
import json
import sys

def train(request):  #a=0下载模型
    file = request.FILES.get('file')
    reader = csv.reader(file.read().decode('utf-8').splitlines())
    adata=[]
    for row in reader:
        adata.append(row)
    # 去重
    print(adata)
    adata1 = []
    adatate = []
    for i in range(len(adata)):
        l = [adata[i][6], adata[i][7]]
        if l not in adatate:
            adatate.append(l)
            adata1.append(adata[i])
    adata = adata1
    print(len(adata))
    # 数据数值化
    adata.pop(0)
    for i in range(len(adata)):
        for j in range(1, len(adata[i]) - 1):
            if adata[i][j] != '':
                adata[i][j] = float(adata[i][j])
            else:
                adata[i][j] = None
        adata[i].pop(0)
    lab = []
    for i in range(len(adata)):
        lab.append(int(adata[i][107]))
        adata[i].pop(107)
    #填补缺失值
    def add_average(l):
        df=pd.DataFrame(l)
        df1=df.fillna(df.mean())
        df1=df1.values.tolist()
        return df1
    adata = add_average(adata)
    # 归一化
    df = pd.DataFrame(adata)
    scaler = StandardScaler()
    scaler.fit(df)
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    adata = df_normalized.values.tolist()
    #生成输入数据
    n = len(adata)
    data = np.zeros([n, 107])  # 用来装样本自变量
    for i in range(n):
        for j in range(107):
            data[i, j] = adata[i][j]
    x_train = data
    y_train = lab
    # 不平衡处理
    def smote(x, y):
        x, y = KMeansSMOTE(cluster_balance_threshold=0.064, random_state=42).fit_resample(x, y)
        return x, y
    x_train, y_train = smote(x_train, y_train)
    import catboost
    # 模型训练
    model = catboost.CatBoostClassifier(verbose=True, learning_rate=0.01).fit(x_train, y_train)
    print('train over')
    #模型保存
    # if a==0:
    model.save_model('model.bin')
    print('save model success')
    return HttpResponse('success')

def download(request):
    # 获取模型文件的路径
    model_path = './model.bin'
    # 检查文件是否存在
    if os.path.exists(model_path):
        # 创建文件响应对象并返回给客户端
        response = FileResponse(open(model_path, 'rb'), as_attachment=True)
        response['Content-Disposition'] = 'attachment; filename="model.bin"'
        return response
    else:
        # 返回错误响应...
        return HttpResponse('Model file not found.')

def predict(request): 
    file = request.FILES.get('file')
    reader = csv.reader(file.read().decode('utf-8').splitlines())
    testdata=[]
    for row in reader:
        testdata.append(row)
    a=len(testdata)# a!=1 多条预测，a=1单条预测
    print(a)
    # print(testdata)
    if a!=1:
        #数值化
        index=[]
        testdata.pop(0)
        for i in range(len(testdata)):
            for j in range(1, len(testdata[i])):
                if testdata[i][j] != '':
                    testdata[i][j] = float(testdata[i][j])
                else:
                    testdata[i][j] = None
            index.append(testdata[i][0])
            testdata[i].pop(0)
        # 填补缺失值
        def add_average(l):
            df = pd.DataFrame(l)
            df1 = df.fillna(df.mean())
            df1 = df1.values.tolist()
            return df1
        testdata = add_average(testdata)
        # 标准化
        df = pd.DataFrame(testdata)
        scaler = StandardScaler()
        scaler.fit(df)
        df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        testdata = df_normalized.values.tolist()
        #生成输入数据
        n = len(testdata)
        data = np.zeros([n, 107])  # 用来装样本自变量
        for i in range(n):
            for j in range(107):
                data[i, j] = testdata[i][j]
        import catboost
        model = catboost.CatBoostClassifier().load_model('model2.bin')
        result = model.predict(data)
        result1=[]
        for i in range(len(result)):
            result1.append(result[i][0])
        typenum = [0, 0, 0, 0, 0, 0]
        for i in range(len(result)):
            typenum[result[i][0]] += 1
        data = {'typenum': np.array(typenum).tolist(), 
                'result1': np.array(result1).tolist(), 
                'index': np.array(index).tolist()}
        json_data = json.dumps(data)
        return JsonResponse(json_data, safe=False)#返回 typenum为各个类别对应数量，result1为列表形式的分类结果，index为每条结果对应的索引
    else:
        adata=[]
        with open('datasets/train_10000.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                adata.append(row)
        f.close()
        # 数值化
        adata.pop(0)
        testdata.pop(0)
        for i in range(len(testdata)):
            for j in range(1, len(testdata[i])):
                if testdata[i][j] != '':
                    testdata[i][j] = float(testdata[i][j])
                else:
                    testdata[i][j] = None
            testdata[i].pop(0)
        for i in range(len(adata)):
            for j in range(1, len(adata[i])):
                if adata[i][j] != '':
                    adata[i][j] = float(adata[i][j])
                else:
                    adata[i][j] = None
            adata[i].pop(0)
        # 填补缺失值
        featureaverage = [0 for i in range(107)]
        nonum = [0 for i in range(107)]
        for i in range(len(adata)):
            for j in range(107):
                if adata[i][j] != None:
                    featureaverage[j] += adata[i][j]
                    nonum[j] += 1
        for i in range(len(featureaverage)):
            featureaverage[i] = featureaverage[i] / nonum[i]
        def add_sameaverage(l):
            for i in range(len(l)):
                for j in range(len(l[i])):
                    if l[i][j] == None:
                        l[i][j] = featureaverage[j]
            return l
        testdata = add_sameaverage(testdata)
        # 标准化
        df = pd.DataFrame(adata)
        scaler = StandardScaler()
        scaler.fit(df)
        df = pd.DataFrame(testdata)
        df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        testdata = df_normalized.values.tolist()
        # 生成输入数据
        n = len(testdata)
        data = np.zeros([n, 107])  # 用来装样本自变量
        for i in range(n):
            for j in range(107):
                data[i, j] = testdata[i][j]
        import catboost
        model = catboost.CatBoostClassifier().load_model('model2.bin')
        result = model.predict(data)
        return HttpResponse(result[0])  #返回数字