#!/usr/bin/env python
# coding: utf-8

# ![](./img/dl_banner.jpg)

# # 使用TensorFlow seq2seq解决时序预估问题
# 
# 在这个notebook中将讲解到使用tensorflow的seq2seq解决时序预测问题的基本做法，其中的代码大家可以当做模板使用在自己的时序场景中，我们会从基本的时间序列预测开始，介绍到实际(带噪声)数据上的建模，大家也可以看到seq2seq模型在这类问题上的显著效果，可以比较好地挖掘出底层的数据分布模式规律。
# 
# ## seq2seq
# **参考资料**：
# - [真正的完全图解Seq2Seq Attention模型](https://zhuanlan.zhihu.com/p/40920384)
# - [Overview - seq2seq - Google](https://google.github.io/seq2seq/)
# - [google seq2seq github](https://github.com/google/seq2seq)
# 
# seq2seq 是一个 Encoder–Decoder 结构的网络，它的输入是一个序列，输出也是一个序列， Encoder 中将一个可变长度的信号序列变为固定长度的向量表达，Decoder 将这个固定长度的向量变成可变长度的目标的信号序列。
# 这个结构最灵活的地方在于输入序列和输出序列的长度是可变的，可以用于翻译，聊天机器人，句法分析，文本摘要等；当然在我们的场景下用于序列预估也是合适的。seq2seq模型的大体结构如下图所示。
# ![](./img/seq2seq.gif)
# ![](./img/seq2seq_7.gif)
# 
# 
# ## 主要内容
# 
# ### 1) <b>[单变量时间序列问题](#session1)</b> 
# 
# ### 2) <b>[多变量时间序列问题](#session2)</b> 
# 
# ### 3) <b>[Seq2seq预估与极端与异常值检测](#session3)</b> 
# 
# ### 4) <b>[基于深度学习时序模型的北京PM2.5预测](#session4)</b> 
# 数据集来源于UCI - https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data

# <a id='session1'></a>
# # 单变量时序预测

# In[1]:


import tensorflow as tf
import numpy as np 
import random
import math
from matplotlib import pyplot as plt
import os
import copy


# In[2]:


x = np.linspace(0, 30, 120)
y = 1.5 * np.sin(x)

l1, = plt.plot(x[:100], y[:100], 'y', label = 'training samples')
l2, = plt.plot(x[100:], y[100:], 'c--', label = 'test samples')
plt.legend(handles = [l1, l2], loc = 'upper left')
plt.show()


# <a id='Session2'></a>
# ## 带噪数据
# 真实世界的数据往往并不是那么干净的，而是带着噪声的数据。我们在上述数据上手动添加上一些噪声，得到以下的数据。

# In[3]:


train_y = y.copy()

noise_factor = 0.45
train_y += np.random.randn(120) * noise_factor

l1, = plt.plot(x[:100], train_y[:100], 'yo', label = 'training samples')
plt.plot(x[:100], y[:100], 'y:')
l2, = plt.plot(x[100:], train_y[100:], 'co', label = 'test samples')
plt.plot(x[100:], y[100:], 'c:')
plt.legend(handles = [l1, l2], loc = 'upper left')
plt.show()


# ## 构建训练样本
# 如果我们想用seq2seq的模型去学习时序数据的规律模式，我们需要构建出合理的训练样本数据，通常我们会以k个时间点的数据作为输入，以之后的k个时间点的数据作为输出，这个长度为2k的窗口可以滑动构建出一批一批的样本。

# In[4]:


input_seq_len = 15
output_seq_len = 20

x = np.linspace(0, 30, 120)
train_data_x = x[:100]

def true_signal(x):
    y = 2 * np.sin(x)
    return y

def noise_func(x, noise_factor = 1):
    return np.random.randn(len(x)) * noise_factor

def generate_y_values(x):
    return true_signal(x) + noise_func(x)

def generate_train_samples(x = train_data_x, batch_size = 10, input_seq_len = input_seq_len, output_seq_len = output_seq_len):

    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size)
    
    input_seq_x = [x[i:(i+input_seq_len)] for i in start_x_idx]
    output_seq_x = [x[(i+input_seq_len):(i+input_seq_len+output_seq_len)] for i in start_x_idx]
    
    input_seq_y = [generate_y_values(x) for x in input_seq_x]
    output_seq_y = [generate_y_values(x) for x in output_seq_x]
    
    #batch_x = np.array([[true_signal()]])
    return np.array(input_seq_y), np.array(output_seq_y)


# In[5]:


input_seq, output_seq = generate_train_samples(batch_size=10)


# In[6]:


# 一个训练样本
l1, = plt.plot(range(15), input_seq[1], 'go', label = 'input sequence for one sample')
l2, = plt.plot(range(15, 35), output_seq[1], 'ro', label = 'output sequence for one sample')
plt.legend(handles = [l1, l2], loc = 'lower left')
plt.show()


# #### 大概给大家画一个示意图，以下是带噪声的数据分布，以及隐藏在数据分布下的(我们希望挖掘的)规律

# In[7]:


results = []
for i in range(100):
    temp = generate_y_values(x)
    results.append(temp)
results = np.array(results)

for i in range(100):
    l1, = plt.plot(results[i].reshape(120, -1), 'co', lw = 0.1, alpha = 0.05, label = 'noisy training data')

l2, = plt.plot(true_signal(x), 'm', label = 'hidden true signal')
plt.legend(handles = [l1, l2], loc = 'lower left')
plt.show()


# ## seq2seq模型初步应用
# 大家在google tensorflow的[这个文件](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py)里可以找到tensorflow的seq2seq的实现，这里把它摘抄到build_model_basic.py中，并做一点小小的修改。

# In[8]:


from build_model_basic import * 

## 训练超参数
learning_rate = 0.01
lambda_l2_reg = 0.003  

## 网络超参数
# 输入序列长度
input_seq_len = 15 
# 输出信号长度
output_seq_len = 20 
# LSTM的隐层神经元个数
hidden_dim = 64 
# 输入维度
input_dim = 1
# 输出维度
output_dim = 1
# 多层lstm的层数
num_stacked_layers = 2 
# 梯度裁剪：防止梯度爆炸
GRADIENT_CLIPPING = 2.5 


# ## 训练

# In[9]:


# 总迭代轮次
total_iteractions = 100
# 一批数据的样本数
batch_size = 16
# dropout的保留率
KEEP_RATE = 0.5
# 存放训练和验证的loss
train_losses = []
val_losses = []

# 产出数据
x = np.linspace(0, 30, 120)
train_data_x = x[:100]

# 构建模型
rnn_model = build_graph(feed_previous=False)
# 模型存储
saver = tf.train.Saver()

# 初始化op
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 在session中训练
    for i in range(total_iteractions):
        batch_input, batch_output = generate_train_samples(batch_size=batch_size)
        # 传入数据
        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t].reshape(-1,input_dim) for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t].reshape(-1,output_dim) for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        print(loss_t)
    # 模型存储
    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./models', 'ts_model'))
        
print("Checkpoint存储在: ", save_path)


# ## 预估

# In[10]:


# 真实结果
test_seq_input = true_signal(train_data_x[-15:])
# 构建模型
rnn_model = build_graph(feed_previous=True)
# 初始化op
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 加载模型
    saver = rnn_model['saver']().restore(sess, os.path.join('./models', 'ts_model'))
    # 传入数据
    feed_dict = {rnn_model['enc_inp'][t]: test_seq_input[t].reshape(1,1) for t in range(input_seq_len)}
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, output_dim]) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
    # 拼接预估结果
    final_preds = np.concatenate(final_preds, axis = 1)


# ## 绘制出预估值

# In[11]:


l1, = plt.plot(range(100), true_signal(train_data_x[:100]), label = 'Training truth')
l2, = plt.plot(range(100, 120), y[100:], 'yo', label = 'Test truth')
l3, = plt.plot(range(100, 120), final_preds.reshape(-1), 'ro', label = 'Test predictions')
plt.legend(handles = [l1, l2, l3], loc = 'lower left')
plt.show()


# <a id='session2'></a>
# # 多变量时间序列问题: 

# ### 多变量序列预估
# 现实生活中我们更多的情况是，有多个输入和输出信号。假设我们想预测未来1天的交通状况; 影响结果的不仅可能是过去的交通状况，还可能是天气，温度，星期几，是否节假日等等。

# In[12]:


x = np.linspace(0, 40, 130)
x1 = 2 * np.sin(x)
x2 = 2 * np.cos(x)

y1 = 1.5*x1**4 - 1.8*x2 - 8
y2 = 1.2*x2**2*x1 + 6*x2 - 6*x1
y3 = 2.2*x1**3 + 2.1*x2**3 - x1*x2

plt.title("Ground truth for 3 input signals")
l1, = plt.plot(y1, 'r', label = 'signal 1')
l2, = plt.plot(y2, 'g', label = 'signal 2')
l3, = plt.plot(y3, 'b', label = 'signal 3')

plt.legend(handles = [l1, l2, l3], loc = 'upper right')

plt.show()


# ### 数据分布模式
# 我们观测到的是3组不同的时间序列，而实际底层的模式是sin和cos两组序列决定的，我们绘图表示一下。

# In[13]:


x = np.linspace(0, 40, 130)
x1 = 2 * np.sin(x)
x2 = 2 * np.cos(x)

y1 = 1.5*x1**4 - 1.8*x2 - 8
y2 = 1.2*x2**2*x1 + 6*x2 - 6*x1
y3 = 2.2*x1**3 + 2.1*x2**3 - x1*x2

plt.title("Discover hidden signals from observed facts")
l1, = plt.plot(y1, 'c:', label = 'input signals (observed facts)')
plt.plot(y2, 'c:')
plt.plot(y3, 'c:')

l4, = plt.plot(2 * x1, 'm', label = 'output signals (hidden signals)')
plt.plot(2 * x2, 'm') # multiplies by 2 just for visualization purpose

plt.legend(handles = [l1, l4], loc = 'upper right')

plt.show()


# ## 构建样本
# 我们来构建更特殊一点的输入输出样本，这里的输入是带噪声的sin和cos函数，而输出是我们观测到的3组序列数据。

# In[14]:


# 输入序列长度
input_seq_len = 15
# 输出序列长度
output_seq_len = 20
# 输入的时间index
x = np.linspace(0, 40, 130)
# 输入数据
train_data_x = x[:110]

# 真实的输出信号序列
def true_output_signals(x):
    x1 = 2 * np.sin(x)
    x2 = 2 * np.cos(x)
    return x1, x2

# 真实的输入信号序列
def true_input_signals(x):
    x1, x2 = true_output_signals(x)
    y1 = 1.6*x1**4 - 2*x2 - 10
    y2 = 1.2*x2**2 * x1 + 2*x2*3 - x1*6
    y3 = 2*x1**3 + 2*x2**3 - x1*x2
    return y1, y2, y3

# 加噪声的函数
def noise_func(x, noise_factor = 2):
    return np.random.randn(len(x)) * noise_factor

# 生成输出样本序列
def generate_samples_for_output(x):
    x1, x2 = true_output_signals(x)
    return x1+noise_func(x1, 0.5),            x2+noise_func(x2, 0.5)

# 生成输入样本序列
def generate_samples_for_input(x):
    y1, y2, y3 = true_input_signals(x)
    return y1+noise_func(y1, 2),            y2+noise_func(y2, 2),            y3+noise_func(y3, 2)

# 生成训练样本
def generate_train_samples(x = train_data_x, batch_size = 10):
    # 起始位置
    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size)
    # 输入序列与输出序列构建
    input_seq_x = [x[i:(i+input_seq_len)] for i in start_x_idx]
    output_seq_x = [x[(i+input_seq_len):(i+input_seq_len+output_seq_len)] for i in start_x_idx]
    input_seq_y = [generate_samples_for_input(x) for x in input_seq_x]
    output_seq_y = [generate_samples_for_output(x) for x in output_seq_x]
    
    ## 转置与维度变更 : (batch_size, time_steps, feature_dims)
    return np.array(input_seq_y).transpose(0, 2, 1), np.array(output_seq_y).transpose(0, 2, 1)


# ### 可视化训练样本
# 取出一个训练样本进行可视化

# In[15]:


input_seq, output_seq = generate_train_samples(batch_size=100)

i1, i2, i3= plt.plot(range(input_seq_len), input_seq[0], 'yo', label = 'input_seqs_one_sample')
o1, o2 = plt.plot(range(input_seq_len,(input_seq_len+output_seq_len)), output_seq[0], 'go', label = 'output_seqs_one_sample')
plt.legend(handles = [i1, o1])
plt.show()


# ## 构建模型

# In[16]:


from build_model_multi_variate import * 

## 训练超参数
learning_rate = 0.01
lambda_l2_reg = 0.003  

## 网络超参数
# 输入序列长度
input_seq_len = 15 
# 输出信号长度
output_seq_len = 20 
# LSTM的隐层神经元个数
hidden_dim = 64 
# 输入维度
input_dim = 3
# 输出维度
output_dim = 2
# 多层lstm的层数
num_stacked_layers = 2 
# 梯度裁剪：防止梯度爆炸
GRADIENT_CLIPPING = 2.5 


# ## 训练模型

# In[19]:


# 总迭代轮次
total_iteractions = 100
# 一批数据的样本数
batch_size = 16
# dropout的保留率
KEEP_RATE = 0.5
# 存放训练和验证的loss
train_losses = []
val_losses = []

# 产出数据
x = np.linspace(0, 40, 130)
train_data_x = x[:110]

# 构建模型
rnn_model = build_graph(feed_previous=False)
# 模型存储器
saver = tf.train.Saver()

# 初始化op
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 迭代与训练
    print("训练损失: ")
    for i in range(total_iteractions):
        batch_input, batch_output = generate_train_samples(batch_size=batch_size)
        # 传入数据
        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        print(loss_t)
        
    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./models', 'multivariate_ts_model'))
        
print("Checkpoint存储于: ", save_path)


# ## 预测

# In[27]:


# 预测输入
test_seq_input = np.array(generate_samples_for_input(train_data_x[-15:])).transpose(1,0)
# 构建模型
rnn_model = build_graph(feed_previous=True)
# 初始化op
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 加载模型
    saver = rnn_model['saver']().restore(sess, os.path.join('./models', 'multivariate_ts_model'))
    # 传入数据
    feed_dict = {rnn_model['enc_inp'][t]: test_seq_input[t].reshape(1, -1) for t in range(input_seq_len)}
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, output_dim], dtype=np.float32) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
    # 拼接预估结果
    final_preds = np.concatenate(final_preds, axis = 1)


# In[31]:


final_preds


# ## 可视化预测结果

# In[36]:


test_seq_input = np.array(generate_samples_for_input(train_data_x[-15:])).transpose(1,0)
test_seq_output = np.array(generate_samples_for_output(train_data_x[-20:])).transpose(1,0)
plt.title("Input sequence, predicted and true output sequences")
i1, i2, i3, = plt.plot(range(15), np.array(true_input_signals(x[95:110])).transpose(1, 0), 'c:', label = 'true input sequence')
p1, p2 = plt.plot(range(15, 35), 4 * final_preds.reshape(20,-1), 'ro', label = 'predicted outputs')
t1, t2 = plt.plot(range(15, 35), 4 * np.array(true_output_signals(x[110:])).transpose(1, 0), 'co', alpha = 0.6, label = 'true outputs')
plt.legend(handles = [i1, p1, t1], loc = 'upper left')
plt.show()


# In[37]:


test_seq_input = np.array(generate_samples_for_input(train_data_x[-15:])).transpose(1,0)
test_seq_output = np.array(generate_samples_for_output(train_data_x[-20:])).transpose(1,0)
plt.title("Predicted and true output sequences")
#i1, i2, i3, = plt.plot(range(15), np.array(true_input_signals(x[95:110])).transpose(1, 0), 'c:', label = 'true input sequence')
p1, p2 = plt.plot(range(15, 35), final_preds.reshape(20,-1), 'ro', label = 'predicted outputs')
t1, t2 = plt.plot(range(15, 35), np.array(true_output_signals(x[110:])).transpose(1, 0), 'co', label = 'true outputs')
plt.legend(handles = [p1, t1], loc = 'upper left')
plt.show()


# <a id='session3'></a>
# # 3.Seq2seq预估与极端与异常值检测
# 
# ## 3.1 样本构建

# In[38]:


# 我们来构建一份数据，包含一些明显的极端值
x = np.linspace(0, 60, 210)
y = 2 * np.sin(x)
num_events_train = 40
num_events_test = 1
extreme_factor = 2
np.random.seed(10)
train_events_x = np.random.choice(range(190), num_events_train)
test_events_x = np.random.choice(range(190, 210), num_events_test)
y[train_events_x] += extreme_factor
y[test_events_x] += extreme_factor

plt.title('No. of traffic v.s. days')

l1, = plt.plot(x[:190], y[:190], 'y', label = 'training true signal')
l2, = plt.plot(x[190:], y[190:], 'c--', label = 'test true signal')

plt.legend(handles = [l1, l2], loc = 'upper left')
plt.show()


# In[39]:


# 输入序列长度
input_seq_len = 15
# 输出序列长度
output_seq_len = 20
# 添加的异常点系数
extreme_factor = 2
# 输入的时间index
x = np.linspace(0, 60, 210)
# 输入数据
train_data_x = x[:190]

# 设置随机种子
np.random.seed(10)
# 训练集中的异常点个数
num_events_train = 40 
train_events_bool = np.zeros(190)
train_events_x = np.random.choice(range(190), num_events_train)
train_events_bool[train_events_x] = 1

# 测试集中的异常点个数
num_events_test = 1 # total number of extreme events in test data 
test_events_bool = np.zeros(20)
test_events_x = np.random.choice(range(20), num_events_test)
test_events_bool[test_events_x] = 1.
                    

# 真实的输出信号序列
def true_signal(x):
    y = 2 * np.sin(x)
    return y

# 加噪声函数
def noise_func(x, noise_factor = 1):
    return np.random.randn(len(x)) * noise_factor

# 带噪的输出结果
def generate_y_values(x, event_bool):
    return true_signal(x) + noise_func(x) + extreme_factor * event_bool

# 生成训练样本
def generate_train_samples(x = train_data_x, batch_size = 10, input_seq_len = input_seq_len, output_seq_len = output_seq_len, train_events_bool = train_events_bool):
    # 起始位置
    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size)
    # 输入序列与输出序列构建
    input_seq_x = [x[i:(i+input_seq_len)] for i in start_x_idx]
    input_seq_event_bool = [train_events_bool[i:(i+input_seq_len)] for i in start_x_idx]
    output_seq_x = [x[(i+input_seq_len):(i+input_seq_len+output_seq_len)] for i in start_x_idx]
    output_seq_event_bool = [train_events_bool[(i+input_seq_len):(i+input_seq_len+output_seq_len)] for i in start_x_idx]
    input_seq_y = [generate_y_values(x, event_bool) for x, event_bool in zip(input_seq_x, input_seq_event_bool)]
    output_seq_y = [generate_y_values(x, event_bool) for x, event_bool in zip(output_seq_x, output_seq_event_bool)]
    
    # 返回，batch_x = np.array([[true_signal()]])
    return np.array(input_seq_y), np.array(output_seq_y), np.array(input_seq_event_bool), np.array(output_seq_event_bool)


# ### 可视化训练样本
# 取出一个训练样本进行可视化

# In[40]:


input_seq, output_seq, input_seq_event_bool, output_seq_event_bool = generate_train_samples(batch_size=10)
l1, = plt.plot(range(15), input_seq[0], 'go', label = 'input sequence for one sample')
l2, = plt.plot(range(15, 35), output_seq[0], 'ro', label = 'output sequence for one sample')
plt.legend(handles = [l1, l2], loc = 'lower left')
plt.show()


# ## 3.2.1 一般的seq2seq模型训练

# In[41]:


from build_model_basic import * 

## 训练超参数
learning_rate = 0.01
lambda_l2_reg = 0.003  

## 网络超参数
# 输入序列长度
input_seq_len = 15 
# 输出信号长度
output_seq_len = 20 
# LSTM的隐层神经元个数
hidden_dim = 64 
# 输入维度
input_dim = 1
# 输出维度
output_dim = 1
# 多层lstm的层数
num_stacked_layers = 2 
# 梯度裁剪：防止梯度爆炸
GRADIENT_CLIPPING = 2.5 


# In[43]:


# 总迭代轮次
total_iteractions = 200
# 一批数据的样本数
batch_size = 16
# dropout的保留率
KEEP_RATE = 0.5
# 存放训练和验证的loss
train_losses = []
val_losses = []

# 产出数据
x = np.linspace(0, 60, 210)
train_data_x = x[:190]

# 构建模型
rnn_model = build_graph(feed_previous=False)
# 模型存储器
saver = tf.train.Saver()

# 初始化op
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 迭代200轮
    for i in range(total_iteractions):
        batch_input, batch_output, input_seq_event_bool, output_seq_event_bool = generate_train_samples(batch_size=batch_size)
        # 传入数据
        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t].reshape(-1,input_dim) for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t].reshape(-1,output_dim) for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        print(loss_t)
    # 存储模型
    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./models', 'univariate_ts_model'))
        
print("Checkpoint存储在: ", save_path)


# ## 预估

# In[44]:


# 测试集输入
test_seq_input = true_signal(train_data_x[-15:])
# 构建模型
rnn_model = build_graph(feed_previous=True)
# 初始化op
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 加载模型
    saver = rnn_model['saver']().restore(sess, os.path.join('./models', 'univariate_ts_model'))
    # 传入数据
    feed_dict = {rnn_model['enc_inp'][t]: test_seq_input[t].reshape(1,1) for t in range(input_seq_len)}
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, output_dim]) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
    # 拼接预估结果
    final_preds = np.concatenate(final_preds, axis = 1)


# ### 可视化预估结果与真实值

# In[45]:


l2, = plt.plot(range(190, 210), y[190:], 'yo', label = 'Test truth')
l3, = plt.plot(range(190, 210), final_preds.reshape(-1), 'ro', label = 'Test predictions')
plt.legend(handles = [l2, l3], loc = 'lower right')
plt.show()


# ## 2. 使用seq2seq处理异常值预测

# In[48]:


from build_model_with_outliers import * 

## 训练超参数
learning_rate = 0.01
lambda_l2_reg = 0.003  

## 网络超参数
# 输入序列长度
input_seq_len = 15 
# 输出信号长度
output_seq_len = 20 
# LSTM的隐层神经元个数
hidden_dim = 64 
# 输入维度
input_dim = 1
# 输出维度
output_dim = 1
# 多层lstm的层数
num_stacked_layers = 2 
# 梯度裁剪：防止梯度爆炸
GRADIENT_CLIPPING = 2.5 


# In[58]:


# 总迭代轮次
total_iteractions = 100
# 一批数据的样本数
batch_size = 16
# dropout的保留率
KEEP_RATE = 0.5
# 存放训练和验证的loss
train_losses = []
val_losses = []

# 产出数据
x = np.linspace(0, 60, 210)
train_data_x = x[:190]

# 构建模型
rnn_model = build_graph(feed_previous=False)
# 模型存储器
saver = tf.train.Saver()

# 初始化op
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 迭代200轮    
    for i in range(total_iteractions):
        batch_input, batch_output, batch_in_event_bool, batch_out_event_bool = generate_train_samples(batch_size=batch_size)
        # 传入数据
        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t].reshape(-1,input_dim) for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t].reshape(-1,output_dim) for t in range(output_seq_len)})
        #feed_dict.update({rnn_model['input_seq_extremes_bool'][t]: batch_in_event_bool[:,t].reshape(-1,1) for t in range(input_seq_len)})
        feed_dict.update({rnn_model['output_seq_extremes_bool'][t]: batch_out_event_bool[:,t].reshape(-1,1) for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        print(loss_t)
    # 存储模型
    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./models', 'univariate_ts_model_eventsBool'))

print("Checkpoint存储在: ", save_path)


# ## 预估

# In[59]:


# 准备数据集
test_seq_input = true_signal(train_data_x[-15:])
test_events_bool_input = train_events_bool[-15:]
test_events_bool_output = test_events_bool.copy()

# 构建模型
rnn_model = build_graph(feed_previous=True)

# 初始化op
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 加载模型
    saver = rnn_model['saver']().restore(sess, os.path.join('./models', 'univariate_ts_model_eventsBool'))
    # 预测
    feed_dict = {rnn_model['enc_inp'][t]: test_seq_input[t].reshape(1,1) for t in range(input_seq_len)}
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, 1]) for t in range(output_seq_len)})
    feed_dict.update({rnn_model['output_seq_extremes_bool'][t]: test_events_bool_output[t].reshape(1,1) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)


# In[60]:


# 预估值与真实值比对
final_preds  = np.array(final_preds).reshape(-1)
l2, = plt.plot(range(190, 210), y[190:], 'yo', label = 'Test truth')
l3, = plt.plot(range(190, 210), final_preds.reshape(-1), 'ro', label = 'Test predictions')
plt.legend(handles = [l2, l3], loc = 'lower right')
plt.show()


# <a id='session4'></a>
# # 4.基于深度学习时序模型的北京PM2.5预测

# 本案例的数据集来源于UCI的[这个地址](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)，我们运用上面讲到的内容，对典型的背景空气质量进行序列预测建模。

# ## 数据读入与探索

# In[61]:


# 导入工具库
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# 读入数据
df = pd.read_csv('./data/PRSA_data_2010.1.1-2014.12.31.csv')
print(df.head())

# 可以绘图的列名
cols_to_plot = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
i = 1

# 绘图
plt.figure(figsize = (10,12))
for col in cols_to_plot:
    plt.subplot(len(cols_to_plot), 1, i)
    plt.plot(df[col])
    plt.title(col, y=0.5, loc='left')
    i += 1
plt.show()


# ## 数据预处理

# In[62]:


## 缺失值填充：填充0 
df.fillna(0, inplace = True)

## 对列'cbwd'进行独热向量编码
temp = pd.get_dummies(df['cbwd'], prefix='cbwd')
df = pd.concat([df, temp], axis = 1)
del df['cbwd'], temp

## 切分训练集和测试集，这里切分了尾部1个月的数据作为测试集
df_train = df.iloc[:(-31*24), :].copy()
df_test = df.iloc[-31*24:, :].copy()

## 取出其中的部分特征列用于建模，你可以自行添加'hour', 'day' 或者 'month'，看看对泛化能力提升是否有帮助
X_train = df_train.loc[:, ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']].values.copy()
X_test = df_test.loc[:, ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv']].values.copy()
y_train = df_train['pm2.5'].values.copy().reshape(-1, 1)
y_test = df_test['pm2.5'].values.copy().reshape(-1, 1)

## 神经网络的输入是需要做预处理(幅度缩放)的，当然，独热向量编码的01列不需要
for i in range(X_train.shape[1]-4):
    temp_mean = X_train[:, i].mean()
    temp_std = X_train[:, i].std()
    X_train[:, i] = (X_train[:, i] - temp_mean) / temp_std
    X_test[:, i] = (X_test[:, i] - temp_mean) / temp_std
    
## 对label做一样的处理
y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std


# ## 数据准备
# 我们吧训练集和测试集都处理成(batch_size, time_step, feature_dim)维度

# In[63]:


# 输入序列长度
input_seq_len = 30
# 输出序列长度
output_seq_len = 5

# 生成训练样本
def generate_train_samples(x = X_train, y = y_train, batch_size = 10, input_seq_len = input_seq_len, output_seq_len = output_seq_len):
    # 起始位置
    total_start_points = len(x) - input_seq_len - output_seq_len
    start_x_idx = np.random.choice(range(total_start_points), batch_size, replace = False)
    # 输入与输出序列
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in start_x_idx]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
    # 最终维度(batch_size, time_steps, feature_dim)
    return input_seq, output_seq

# 生成测试样本
def generate_test_samples(x = X_test, y = y_test, input_seq_len = input_seq_len, output_seq_len = output_seq_len):
    # 起始位置
    total_samples = x.shape[0]
    # 输入与输出序列
    input_batch_idxs = [list(range(i, i+input_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    input_seq = np.take(x, input_batch_idxs, axis = 0)
    output_batch_idxs = [list(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in range((total_samples-input_seq_len-output_seq_len))]
    output_seq = np.take(y, output_batch_idxs, axis = 0)
    # 返回
    return input_seq, output_seq


# In[64]:


# 生成1个batch样本
x, y = generate_train_samples()
print(x.shape, y.shape)


# In[65]:


# 测试样本
test_x, test_y = generate_test_samples()
print(test_x.shape, test_y.shape)


# ## 构建模型

# In[66]:


from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
import tensorflow as tf
import copy
import os

## 网络训练超参数
learning_rate = 0.01
lambda_l2_reg = 0.003  

## 网络结构超参数
# 输入序列长度
input_seq_len = input_seq_len
# 输出序列长度
output_seq_len = output_seq_len
# LSTM隐层维度大小
hidden_dim = 64 
# 输入的维度
input_dim = X_train.shape[1]
# 输出的维度
output_dim = y_train.shape[1]
# 多层LSTM的层数
num_stacked_layers = 2 
# 梯度裁剪：防止梯度爆炸
GRADIENT_CLIPPING = 2.5 

def build_graph(feed_previous = False):
    
    tf.reset_default_graph()
    
    global_step = tf.Variable(
                  initial_value=0,
                  name="global_step",
                  trainable=False,
                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    
    weights = {
        'out': tf.get_variable('Weights_out', \
                               shape = [hidden_dim, output_dim], \
                               dtype = tf.float32, \
                               initializer = tf.truncated_normal_initializer()),
    }
    biases = {
        'out': tf.get_variable('Biases_out', \
                               shape = [output_dim], \
                               dtype = tf.float32, \
                               initializer = tf.constant_initializer(0.)),
    }
                                          
    with tf.variable_scope('Seq2seq'):
        # 编码 Encoder: 输入
        enc_inp = [
            tf.placeholder(tf.float32, shape=(None, input_dim), name="inp_{}".format(t))
               for t in range(input_seq_len)
        ]

        # 解码 Decoder: 输出
        target_seq = [
            tf.placeholder(tf.float32, shape=(None, output_dim), name="y".format(t))
              for t in range(output_seq_len)
        ]

        # Give a "GO" token to the decoder. 
        # If dec_inp are fed into decoder as inputs, this is 'guided' training; otherwise only the 
        # first element will be fed as decoder input which is then 'un-guided'
        dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]

        with tf.variable_scope('LSTMCell'): 
            cells = []
            for i in range(num_stacked_layers):
                with tf.variable_scope('RNN_{}'.format(i)):
                    cells.append(tf.contrib.rnn.LSTMCell(hidden_dim))
            cell = tf.contrib.rnn.MultiRNNCell(cells)
         
        def _rnn_decoder(decoder_inputs,
                        initial_state,
                        cell,
                        loop_function=None,
                        scope=None):
          """RNN decoder for the sequence-to-sequence model.
          Args:
            decoder_inputs: A list of 2D Tensors [batch_size x input_size].
            initial_state: 2D Tensor with shape [batch_size x cell.state_size].
            cell: rnn_cell.RNNCell defining the cell function and size.
            loop_function: If not None, this function will be applied to the i-th output
              in order to generate the i+1-st input, and decoder_inputs will be ignored,
              except for the first element ("GO" symbol). This can be used for decoding,
              but also for training to emulate http://arxiv.org/abs/1506.03099.
              Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
            scope: VariableScope for the created subgraph; defaults to "rnn_decoder".
          Returns:
            A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing generated outputs.
              state: The state of each cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
                (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                 states can be the same. They are different for LSTM cells though.)
          """
          with variable_scope.variable_scope(scope or "rnn_decoder"):
            state = initial_state
            outputs = []
            prev = None
            for i, inp in enumerate(decoder_inputs):
              if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                  inp = loop_function(prev, i)
              if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
              output, state = cell(inp, state)
              outputs.append(output)
              if loop_function is not None:
                prev = output
          return outputs, state

        def _basic_rnn_seq2seq(encoder_inputs,
                              decoder_inputs,
                              cell,
                              feed_previous,
                              dtype=dtypes.float32,
                              scope=None):
          """Basic RNN sequence-to-sequence model.
          This model first runs an RNN to encode encoder_inputs into a state vector,
          then runs decoder, initialized with the last encoder state, on decoder_inputs.
          Encoder and decoder use the same RNN cell type, but don't share parameters.
          Args:
            encoder_inputs: A list of 2D Tensors [batch_size x input_size].
            decoder_inputs: A list of 2D Tensors [batch_size x input_size].
            feed_previous: Boolean; if True, only the first of decoder_inputs will be
              used (the "GO" symbol), all other inputs will be generated by the previous 
              decoder output using _loop_function below. If False, decoder_inputs are used 
              as given (the standard decoder case).
            dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
            scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".
          Returns:
            A tuple of the form (outputs, state), where:
              outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
              state: The state of each decoder cell in the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
          """
          with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
            enc_cell = copy.deepcopy(cell)
            _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
            if feed_previous:
                return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
            else:
                return _rnn_decoder(decoder_inputs, enc_state, cell)

        def _loop_function(prev, _):
          '''Naive implementation of loop function for _rnn_decoder. Transform prev from 
          dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
          used as decoder input of next time step '''
          return tf.matmul(prev, weights['out']) + biases['out']
        
        dec_outputs, dec_memory = _basic_rnn_seq2seq(
            enc_inp, 
            dec_inp, 
            cell, 
            feed_previous = feed_previous
        )

        reshaped_outputs = [tf.matmul(i, weights['out']) + biases['out'] for i in dec_outputs]
        
    # 训练的loss和optimizer
    with tf.variable_scope('Loss'):
        # L2 loss
        output_loss = 0
        for _y, _Y in zip(reshaped_outputs, target_seq):
            output_loss += tf.reduce_mean(tf.pow(_y - _Y, 2))

        # L2 正则化
        reg_loss = 0
        for tf_var in tf.trainable_variables():
            if 'Biases_' in tf_var.name or 'Weights_' in tf_var.name:
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))

        loss = output_loss + lambda_l2_reg * reg_loss

    with tf.variable_scope('Optimizer'):
        optimizer = tf.contrib.layers.optimize_loss(
                loss=loss,
                learning_rate=learning_rate,
                global_step=global_step,
                optimizer='Adam',
                clip_gradients=GRADIENT_CLIPPING)
        
    saver = tf.train.Saver
    
    return dict(
        enc_inp = enc_inp, 
        target_seq = target_seq, 
        train_op = optimizer, 
        loss=loss,
        saver = saver, 
        reshaped_outputs = reshaped_outputs,
        )


# ## 训练模型

# In[67]:


# 总迭代轮次
total_iteractions = 100
# 一批数据的样本数
batch_size = 16
# dropout的保留率
KEEP_RATE = 0.5
# 存放训练和验证的loss
train_losses = []
val_losses = []

# 产出数据
x = np.linspace(0, 40, 130)
train_data_x = x[:110]

# 构建模型
rnn_model = build_graph(feed_previous=False)
# 模型存储器
saver = tf.train.Saver()
# 初始化op
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    # 迭代200轮
    print("训练损失: ")
    for i in range(total_iteractions):
        batch_input, batch_output = generate_train_samples(batch_size=batch_size)
        # 传入数据
        feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t] for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t] for t in range(output_seq_len)})
        _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
        print(loss_t)
    # 存储模型
    temp_saver = rnn_model['saver']()
    save_path = temp_saver.save(sess, os.path.join('./models', 'multivariate_ts_pollution_case'))
        
print("Checkpoint存储在: ", save_path)


# ## 预测
# 注意这里的batch prediction和之前有一些不一样

# In[69]:


rnn_model = build_graph(feed_previous=True)

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    
    saver = rnn_model['saver']().restore(sess,  os.path.join('./models', 'multivariate_ts_pollution_case'))
    # batch prediction
    feed_dict = {rnn_model['enc_inp'][t]: test_x[:, t, :] for t in range(input_seq_len)} 
    feed_dict.update({rnn_model['target_seq'][t]: np.zeros([test_x.shape[0], output_dim], dtype=np.float32) for t in range(output_seq_len)})
    final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
    
    final_preds = [np.expand_dims(pred, 1) for pred in final_preds]
    final_preds = np.concatenate(final_preds, axis = 1)
    print("Test mse is: ", np.mean((final_preds - test_y)**2))


# In[70]:


# 拼接出答案
test_y_expand = np.concatenate([test_y[i].reshape(-1) for i in range(0, test_y.shape[0], 5)], axis = 0)
final_preds_expand = np.concatenate([final_preds[i].reshape(-1) for i in range(0, final_preds.shape[0], 5)], axis = 0)


# In[72]:


# 绘制出预估值和真实值
plt.rcParams['figure.figsize'] = (16, 8)
plt.plot(final_preds_expand, color = 'orange', label = 'predicted')
plt.plot(test_y_expand, color = 'blue', label = 'actual')
plt.title("test data - one month")
plt.legend(loc="upper left")
plt.show()


# ### 版权归 © 稀牛学院 所有 保留所有权利
# ![](./img/xiniu_neteasy.png)

# In[ ]:




