#!/usr/bin/env python
# coding: utf-8

# ![](./img/dl_banner.jpg)

# # 时间序列预测与简单神经网络预估
# #### \[稀牛学院 x 网易云课程\]《深度学习工程师(实战)》课程资料 by [@寒小阳](https://blog.csdn.net/han_xiaoyang)

# In[1]:


get_ipython().system('pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas_datareader seaborn')


# In[2]:


# 基础工具库
get_ipython().run_line_magic('matplotlib', 'inline')
import re
import os
import sys
import datetime
import itertools
import math 


# In[3]:


# 数据分析与处理
import pandas as pd
import pandas_datareader.data as web
import numpy as np
# prevent crazy long pandas prints
pd.options.display.max_columns = 16
pd.options.display.max_rows = 16
pd.set_option('display.float_format', lambda x: '%.5f' % x)
np.set_printoptions(precision=5, suppress=True)

# 过滤warnings
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# 可视化工具库
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# In[5]:


# 特征工程与建模
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import plot_model 


# In[ ]:





# In[6]:


# 设定随机种子
np.random.seed(7)

# 导入数据集
df = pd.read_csv('./data/passengers.csv', sep=';', parse_dates=True, index_col=0)
data = df.values

# 对数据进行类型转换
data = data.astype('float32')

# 训练集测试集切分
train = data[0:120, :]   # length 120
test = data[120:, :]     # length 24


# In[8]:


# 滑窗数据构建函数
def prepare_data(data, lags=1):
    """
    Create lagged data from an input time series
    """
    X, y = [], []
    for row in range(len(data) - lags - 1):
        a = data[row:(row + lags), 0]
        X.append(a)
        y.append(data[row + lags, 0])
    return np.array(X), np.array(y)


# In[9]:


lags = 1
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)
y_true = y_test     # due to naming convention


# 数据的最终形态是这样的
# <pre>
# X       y
# 112     118
# 118     132
# 132     129
# 129     121
# 121     135
# </pre>
# 可以看到，我们总是用前一天的数据去预测下一天的数据

# In[11]:


# 我们把训练序列和测试序列做一个可视化
plt.rcParams['figure.figsize'] = (16, 8)
plt.plot(y_train, label='Original Data | y or t+1', color='#006699')
plt.plot(X_train, label='Lagged Data | X or t', color='orange')
plt.legend(loc='upper left')
plt.title('One Period Lagged Data')
plt.show()


# In[12]:


# 构建简单的多层感知器
mdl = Sequential()
mdl.add(Dense(3, input_dim=lags, activation='relu'))
mdl.add(Dense(1))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=200, batch_size=2, verbose=2)


# In[13]:


# 评估模型效果
train_score = mdl.evaluate(X_train, y_train, verbose=0)
print('模型在训练集上的效果: {:.2f} MSE ({:.2f} RMSE)'.format(train_score, math.sqrt(train_score)))
test_score = mdl.evaluate(X_test, y_test, verbose=0)
print('模型在测试集上的效果: {:.2f} MSE ({:.2f} RMSE)'.format(test_score, math.sqrt(test_score)))


# In[14]:


# 预估训练集
train_predict = mdl.predict(X_train)
test_predict = mdl.predict(X_test)

# 平移训练集以便比对
train_predict_plot = np.empty_like(data)
train_predict_plot[:, :] = np.nan
train_predict_plot[lags: len(train_predict) + lags, :] = train_predict

# 平移测试集以便比对
test_predict_plot = np.empty_like(data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(lags*2)+1:len(data)-1, :] = test_predict

# 绘图与预测
plt.plot(data, label='Observed', color='#006699');
plt.plot(train_predict_plot, label='Prediction for Train Set', color='#006699', alpha=0.5);
plt.plot(test_predict_plot, label='Prediction for Test Set', color='#ff0066');
plt.legend(loc='best');
plt.title('Neural Network')
plt.show()


# In[15]:


mse = ((y_test.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
plt.title('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
plt.plot(y_test.reshape(-1, 1), label='Observed', color='#006699')
plt.plot(test_predict.reshape(-1, 1), label='Prediction', color='#ff0066')
plt.legend(loc='best');
plt.show()


# In[ ]:





# In[16]:


# 滑窗预估：用前n个点预估下1个点
lags = 3
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)

# plot the created data
plt.plot(y_train, label='Original Data | y or t+1', color='#006699')
plt.plot(X_train, label='Lagged Data', color='orange')
plt.legend(loc='best')
plt.title('Three Period Lagged Data')
plt.show()


# In[17]:


# 构建多层感知器
mdl = Sequential()
mdl.add(Dense(4, input_dim=lags, activation='relu'))
mdl.add(Dense(8, activation='relu'))
mdl.add(Dense(1))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=400, batch_size=2, verbose=2)


# In[18]:


# 评估模型效果
train_score = mdl.evaluate(X_train, y_train, verbose=0)
print('模型在训练集上的效果: {:.2f} MSE ({:.2f} RMSE)'.format(train_score, math.sqrt(train_score)))
test_score = mdl.evaluate(X_test, y_test, verbose=0)
print('模型在测试集上的效果: {:.2f} MSE ({:.2f} RMSE)'.format(test_score, math.sqrt(test_score)))


# In[19]:


# 预估训练集
train_predict = mdl.predict(X_train)
test_predict = mdl.predict(X_test)

# 平移训练集以便比对
train_predict_plot = np.empty_like(data)
train_predict_plot[:, :] = np.nan
train_predict_plot[lags: len(train_predict) + lags, :] = train_predict

# 平移测试集以便比对
test_predict_plot = np.empty_like(data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict)+(lags*2)+1:len(data)-1, :] = test_predict

# 绘图与预测
plt.plot(data, label='Observed', color='#006699');
plt.plot(train_predict_plot, label='Prediction for Train Set', color='#006699', alpha=0.5);
plt.plot(test_predict_plot, label='Prediction for Test Set', color='#ff0066');
plt.legend(loc='best');
plt.title('Neural Network')
plt.show()


# In[20]:


y_test


# In[21]:


mse = ((y_test.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
plt.title('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
plt.plot(y_test.reshape(-1, 1), label='Observed', color='#006699')
plt.plot(test_predict.reshape(-1, 1), label='Prediction', color='#ff0066')
plt.legend(loc='upper left');
plt.show()


# In[ ]:


### LSTM


# In[33]:


# 设定随机种子
np.random.seed(1)

# 加载数据集
df = pd.read_csv('./data/passengers.csv', sep=';', parse_dates=True, index_col=0)
data = df.values
data = data.astype('float32')

# 数据幅度缩放(预处理)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data)

# 数据切分
train = dataset[0:120, :]
test = dataset[120:, :]

# 构建训练集
lags = 3
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)

# 对数据reshape变成[samples, time steps, features]的维度
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# In[23]:


# 构建LSTM并训练
mdl = Sequential()
mdl.add(Dense(3, input_shape=(1, lags), activation='relu'))
mdl.add(LSTM(6, activation='relu'))
mdl.add(Dense(1, activation='relu'))
mdl.compile(loss='mean_squared_error', optimizer='adam')
mdl.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)


# In[34]:


from sklearn.metrics import mean_squared_error
# 预估
train_predict = mdl.predict(X_train)
test_predict = mdl.predict(X_test)

# 反变换
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# 结果评估
train_score = math.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
print('模型在训练集上的效果: {:.2f} RMSE'.format(train_score))
test_score = math.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
print('模型在测试集上的效果: {:.2f} RMSE'.format(test_score))


# ### 版权归 © 稀牛学院 所有 保留所有权利
# ![](./img/xiniu_neteasy.png)
