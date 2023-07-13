import numpy as np
import csv
from math import sqrt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout,Conv1D,MaxPooling1D
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler,TomekLinks
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

#读取数据
adata=[]
bdata=[]
with open('./data/validate_1000.csv', 'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        adata.append(row)
f.close()
with open('./data/test_2000_x.csv', 'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        bdata.append(row)
f.close()
#数据数值化
adata.pop(0)
bdata.pop(0)
for i in range(len(adata)):
    for j in range(1,len(adata[i])-1):
        if adata[i][j]!='':
            adata[i][j]=float(adata[i][j])
        else:
            adata[i][j]=None
    adata[i].pop(0)
lab=[]
for i in range(len(adata)):
    lab.append(int(adata[i][107]))
    adata[i].pop(107)
labt=[]
for i in range(len(bdata)):
    for j in range(1,len(bdata[i])-1):
        if bdata[i][j]!='':
            bdata[i][j]=float(bdata[i][j])
        else:
            bdata[i][j]=None
    bdata[i].pop(0)
for i in range(len(bdata)):
    labt.append(int(bdata[i][107]))
    bdata[i].pop(107)
#填补
#各个类别各个指标的均值
classdata=[[],[],[],[],[],[]]
for i in range(len(adata)):
    if lab[i]==0:
        classdata[0].append(adata[i])
    if lab[i]==1:
        classdata[1].append(adata[i])
    if lab[i]==2:
        classdata[2].append(adata[i])
    if lab[i]==3:
        classdata[3].append(adata[i])
    if lab[i]==4:
        classdata[4].append(adata[i])
    if lab[i]==5:
        classdata[5].append(adata[i])
classavedata=[]
for i in range(len(classdata)):
    l=[0 for i in range(107)]
    ln=[0 for i in range(107)]
    for j in range(len(classdata[i])):
        for t in range(107):
            if classdata[i][j][t]!=None:
                l[t]+=classdata[i][j][t]
                ln[t]+=1
    for j in range(len(l)):
        l[j]=l[j]/ln[j]
    classavedata.append(l)
class addvalue():
    def add_average(l):
        df=pd.DataFrame(l)
        df1=df.fillna(df.mean())
        df1=df1.values.tolist()
        return df1
    def add_front(l):
        df = pd.DataFrame(l)
        df1=df.fillna(axis=0,method='ffill')
        df1 = df1.values.tolist()
        return df1
    def add_behind(l):
        df = pd.DataFrame(l)
        df1=df.fillna(axis=1,method='ffill')
        df1 = df1.values.tolist()
        return df1
    def add_x(l,x):
        df = pd.DataFrame(l)
        df1 = df.fillna(x)
        df1 = df1.values.tolist()
        return df1
    def add_lave(l):
        for i in range(len(l)):
            for j in range(len(l[i])):
                if l[i][j]==None:
                    type=lab[i]
                    l[i][j]=classavedata[type][j]
        return l
adata=addvalue.add_average(adata)
bdata=addvalue.add_average(bdata)
#归一化
df=pd.DataFrame(adata)
scaler = MinMaxScaler()
scaler.fit(df)
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
adata=df_normalized.values.tolist()
df=pd.DataFrame(bdata)
test_df_normalized = pd.DataFrame(scaler.transform(df), columns=df.columns)
bdata=test_df_normalized.values.tolist()
n = len(adata)
data = np.zeros([n,107])  # 用来装样本自变量
for i in range(n):
    for j in range(107):
        data[i,j]=adata[i][j]
nt=len(bdata)
datab = np.zeros([nt,107])  # 用来装样本自变量
for i in range(nt):
    for j in range(107):
        datab[i,j]=bdata[i][j]
#one-hot
def to_one_hot(labels, dimension=6):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
# 数据拆分,训练集、测试集
x_train=data
x_test=datab
y_train=lab
y_test=labt
#不平衡处理
class unbalancedo():
    def underbanlance(x,y):
        RUS = RandomUnderSampler(random_state=0)
        x, y = RUS.fit_resample(x, y)
        return x,y
    def overbanlance(x,y):
        RUS = RandomOverSampler(random_state=0)
        x, y = RUS.fit_resample(x, y)
        return x,y
    def smote(x,y):
        x, y = SMOTE(random_state=100).fit_resample(x, y)
    def Tome(x,y):
        x,y=TomekLinks().fit_resample(x, y)
#decision tree
from sklearn.ensemble import RandomForestClassifier
# 模型训练
model = RandomForestClassifier(n_jobs=-1,n_estimators=3000).fit(x_train,y_train)
# 评价
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
#重要性
importances = model.feature_importances_
feat_labels = [i for i in range(107)]
indices = np.argsort(importances)[::-1] #[::-1]表示将各指标按权重大小进行排序输出
indeximportance=[]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
for f in range(x_train.shape[1]):
    print(feat_labels[indices[f]])
    indeximportance.append(feat_labels[indices[f]])
indeximportance=indeximportance[0:25]
nx_train=np.zeros([n,25])
nx_test=np.zeros([nt,25])
for i in range(n):
    for j in range(len(indeximportance)):
        nx_train[i,j]=x_train[i][indeximportance[j]]
for i in range(nt):
    for j in range(len(indeximportance)):
        nx_test[i,j]=x_test[i][indeximportance[j]]
x_train=nx_train
x_test=nx_test

#one-hot
def to_one_hot(labels, dimension=6):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
one_hot_train_labels = to_one_hot(lab)
one_hot_train_labelsb = to_one_hot(labt)
# 数据拆分,训练集、测试集
y_train=one_hot_train_labels
y_test=one_hot_train_labelsb
#x_train,x_test,y_train,y_test = train_test_split(data, one_hot_train_labels, test_size=0.2, random_state=1)
gpus = tf.config.experimental.list_physical_devices('GPU')
from keras.models import Sequential
from keras.layers import Conv2D, Dense, AveragePooling2D, Flatten, BatchNormalization
from keras.optimizers import Adam

#x_train,y_train=unbalancedo.smote(x_train,y_train)
#reshape
X_train = x_train.astype('float32').reshape(len(x_train),5,5,1)
X_test = x_test.astype('float32').reshape(len(x_test),5,5,1)
#因为类别不是从0开始编号，所以进行one-hot编码时减1

model = Sequential()
model.add(Conv2D(12, 6, strides=1, padding='same', input_shape=(5, 5, 1), activation='relu'))
model.add(AveragePooling2D(3, 2, padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(9, 9, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(12, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.25))

model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(6, activation='softmax'))

from keras import backend as K
def get_weight(weights):
    def mycrossentropy(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.zeros_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.ones_like(y_pred))
        loss = (1-weights)*K.binary_crossentropy(y_true, y_pred)*pt_1+weights*K.binary_crossentropy(y_true, y_pred)*pt_0
        return loss
    return mycrossentropy
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
weights=np.array([2,2,2,1,1,1])
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy']) #'categorical_crossentropy'
print('Training')
result = model.fit(X_train, y_train, epochs=300, batch_size=64, validation_split=0.05, shuffle=True)
print('Testing')
from sklearn.metrics import classification_report
y_pred=model.predict(X_test)
y_pred = np.rint(y_pred)
print(classification_report(y_test, y_pred))
#图像
import matplotlib.pyplot as plt
N=300
plt.plot(np.arange(0,N),result.history["loss"],label ="train_loss", c='k', linestyle='--')
plt.plot(np.arange(0,N),result.history["accuracy"],label="train_acc",c='k', linestyle='-')
plt.title("loss and accuracy")
plt.xlabel("epoch")
plt.ylabel("loss/acc")
plt.legend(loc="best")
plt.show()