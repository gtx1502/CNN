# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import random
from PIL import Image

sess = tf.InteractiveSession()

train_dir = 'TRAIN'  # 训练集 0-200
val_dir = 'VAL'  # 验证集 201-255
# test_dir=''

model_path = "/Users/gtx/PycharmProjects/CNN/model1.ckpt"


def train():
    # 训练
    r = 201 * 14 / batch_size + 1
    count = 0
    for i in range(200):
        for j in range(r):
            batch = get_batch(j)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})
        print "%d - 正确率: %g" % (i, accuracy.eval(feed_dict={x: getVal()[0], y_: getVal()[1], keep_prob: 1.0}))
        if float(accuracy.eval(feed_dict={x: getVal()[0], y_: getVal()[1], keep_prob: 1.0})) > 0.987:
            count += 1
        else:
            count = 0
        if count > 8:
            break
    # 训练完成之后，在验证集验证模型分类正确率
    print "验证集正确率: %g" % accuracy.eval(feed_dict={x: getVal()[0], y_: getVal()[1], keep_prob: 1.0})


def get_files(fileName):
    data = []
    count = 0
    for train_folder in os.listdir(fileName):
        for picture in os.listdir(fileName + '/' + train_folder):
            image = Image.open(fileName + '/' + train_folder + '/' + picture)
            lab = np.zeros(14);
            lab[count] = 1;
            pic = []
            for i in range(28):
                for j in range(28):
                    t = image.getpixel((j, i))
                    if t == 255:
                        pic.append(0)
                    else:
                        pic.append(1)
            data.append([pic, lab])
        count += 1
    random.shuffle(data)
    temp = np.array(data)
    temp = temp.transpose()
    return temp


train_data = get_files(train_dir)
val_data = get_files(val_dir)
# test_data=get_files(test_dir)

batch_size = 40


# 数组【任意值，第i个块】
def get_batch(i):
    return train_data[:, i * batch_size:(i + 1) * batch_size].tolist()


def getVal():
    return val_data[:55 * 14:].tolist()


'''def getTest(i):
    return test_data[::].tolist()'''


# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积函数
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 使用2*2的max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# x是特征，y_是真实label。将图片数据从1D转为2D。使用tensor的变形函数tf.reshape
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 14])
x_image = tf.reshape(x, [-1, 28, 28, 1])

########设计卷积神经网络########
# 第一层卷积
# 卷积核尺寸为5*5,1个颜色通道，32个不同的卷积核
W_conv1 = weight_variable([5, 5, 1, 32])
# 用conv2d函数进行卷积操作，加上偏置
b_conv1 = bias_variable([32])
# 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 对卷积的输出结果进行池化操作
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积（和第一层大致相同，卷积核为64，这一层卷积会提取64种特征）
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层。隐含节点数1024。使用ReLU激活函数
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了防止过拟合，在输出层之前加Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层。
W_fc2 = weight_variable([1024, 14])
b_fc2 = bias_variable([14])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

########模型训练设置########
# 定义loss function为cross entropy，优化器使用Adam，并给予一个比较小的学习速率
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(6 * 1e-4).minimize(cross_entropy)

# 定义评测准确率的操作
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

########开始训练过程########

saver = tf.train.Saver()
# 初始化所有参数
tf.initialize_all_variables().run()

'''train()
save_path = saver.save(sess, model_path)
print "Model saved in file: %s" % save_path'''

load_path = saver.restore(sess, model_path)
print "验证集正确率: %g" % accuracy.eval(feed_dict={x: getVal()[0], y_: getVal()[1], keep_prob: 1.0})
# print "测试集正确率: %g" % accuracy.eval(feed_dict={x: getTest()[0], y_: getTest()[1], keep_prob: 1.0})
