import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import tensorflow as tf

def predictmodel(file,a): #a=0 多条预测，a！=0单条预测
    if a==0:
        testdata = []
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                testdata.append(row)
        f.close()
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
        return typenum,result1,index  #返回 typenum为各个类别对应数量，result1为列表形式的分类结果，index为每条结果对应的索引
    else:
        adata=[]
        with open('训练集/train_10000.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                adata.append(row)
        f.close()
        testdata = []
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                testdata.append(row)
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
        return result[0]  #返回数字

file='测试集/test_2000_x.csv'
print(predictmodel(file,0))