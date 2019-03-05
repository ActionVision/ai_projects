#!/usr/bin/env python
# coding: utf-8

# ![](./graphs/dl_banner.jpg)

# # Tensorboard可视化
# #### \[稀牛学院 x 网易云课程\]《深度学习工程师(实战)》课程资料 by [@寒小阳](https://blog.csdn.net/han_xiaoyang)
# 
# Tensorboard是一个非常好用的可视化工具，可以配合tensorflow一起去完成神经网络训练，在网络训练过程中，可以实时对训练中间状态进行可视化观察。
# ![](./graphs/Tensorboard.png)

# ## 1.导入工具库

# In[1]:


from __future__ import print_function
import tensorflow as tf


# ## 2.设定超参数
# #### \[稀牛学院 x 网易云课程\]《深度学习工程师(实战)》课程资料 by [@寒小阳](https://blog.csdn.net/han_xiaoyang)

# In[2]:


# 训练参数
learning_rate = 0.01 # 学习率
training_epochs = 25 # 总迭代轮次
batch_size = 100 # 一批数据大小
display_step = 1 # 信息展示间隔频度
logs_path = '/tmp/tensorflow_logs/' # 日志存储地址

# 网络参数
n_hidden_1 = 256 # 第1个隐层神经元个数
n_hidden_2 = 256 # 第2个隐层神经元个数
n_input = 784 # MNIST数据输入(28*28=784)
n_classes = 10 # MNIST总共有0-9这10个类别

# 占位符
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

# 变量
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W3')
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'b3': tf.Variable(tf.random_normal([n_classes]), name='b3')
}


# ## 3.准备数据

# In[3]:


# 导入 MNIST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# ## 4.构建模型
# #### \[稀牛学院 x 网易云课程\]《深度学习工程师(实战)》课程资料 by [@寒小阳](https://blog.csdn.net/han_xiaoyang)

# In[4]:


# 构建模型
def multilayer_perceptron(x, weights, biases):
    # WX+b再通过非线性变换
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # 为了在Tensorboard中做可视化，summary一下
    tf.summary.histogram("relu1", layer_1)
    # WX+b再通过非线性变换
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # 为了在Tensorboard中做可视化，再次summary一下
    tf.summary.histogram("relu2", layer_2)
    # 全连接层
    out_layer = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    return out_layer


# ## 5.计算损失与优化
# #### \[稀牛学院 x 网易云课程\]《深度学习工程师(实战)》课程资料 by [@寒小阳](https://blog.csdn.net/han_xiaoyang)

# In[5]:


# 把不同的op分到不同scope里，这样可视化会更清晰
with tf.name_scope('Model'):
    # 构建多层感知器
    pred = multilayer_perceptron(x, weights, biases)

with tf.name_scope('Loss'):
    # Softmax交叉熵损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

with tf.name_scope('SGD'):
    # 梯度下降
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # gradient/梯度
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)

with tf.name_scope('Accuracy'):
    # 准确率
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    
    
init = tf.global_variables_initializer()

# 构建loss的summary（是一个标量，用scalar）
tf.summary.scalar("loss", loss)
# 构建accuracy的summary（是一个标量，用scalar）
tf.summary.scalar("accuracy", acc)
# 构建变量的summary（用histogram）
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
# 构建gradient的summary（用histogram）
for grad, var in grads:
    tf.summary.histogram(var.name + '/gradient', grad)

# 注意这一步，Merge所有的summaries
merged_summary_op = tf.summary.merge_all()


# ## 6.在session当中完成计算图计算(损失计算与优化、参数更新迭代)
# #### \[稀牛学院 x 网易云课程\]《深度学习工程师(实战)》课程资料 by [@寒小阳](https://blog.csdn.net/han_xiaoyang)

# In[6]:


# 在session中开始训练
with tf.Session() as sess:

    # 初始化所有变量
    sess.run(init)

    # 要把log写出去，以便Tensorboard可视化
    summary_writer = tf.summary.FileWriter(logs_path,
                                            graph=tf.get_default_graph())

    # 训练的迭代过程
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历所有batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c, summary = sess.run([apply_grads, loss, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # 每一轮写入日志
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # 计算平均损失
            avg_cost += c / total_batch
        # 打印中间结果展示
        if (epoch+1) % display_step == 0:
            print("第%04d轮" % (epoch+1), "当前损失为", "{:.9f}".format(avg_cost))

    print("训练完成！")

    # 测试集准确率
    print("准确率:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("Run the command line:\n"           "--> tensorboard --logdir=/tmp/tensorflow_logs "           "\nThen open http://0.0.0.0:6006/ into your web browser")


# ![](./graphs/Tensorboard1.png)
# ![](./graphs/Tensorboard2.png)
# ![](./graphs/Tensorboard3.png)
# ![](./graphs/Tensorboard4.png)

# ### 版权归 © 稀牛学院 所有 保留所有权利
# ![](./graphs/xiniu_neteasy.png)

# In[ ]:




