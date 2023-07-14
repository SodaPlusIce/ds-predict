#输入：file:文件路径，modelfile:文件保存路径，a是否下载模型
#输出：无输出，a=0直接保存模型
import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from imblearn.over_sampling import RandomOverSampler, KMeansSMOTE

def trainmodel(file,modelfile,a):  #a=0下载模型
    adata = []
    with open(file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            adata.append(row)
    f.close()
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
    #模型保存
    if a==0:
        model.save_model(modelfile+'model.bin')
file='./训练集/train_10000.csv'
modelfile=''
trainmodel(file,modelfile,1)