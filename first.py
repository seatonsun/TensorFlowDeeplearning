import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time

np.random.seed(1)


# 利用feed_dict来改变x的值


def linear_function():
    """
    实现一个线性功能：
        初始化W，类型为tensor的随机变量，维度为(4,3)
        初始化X，类型为tensor的随机变量，维度为(3,1)
        初始化b，类型为tensor的随机变量，维度为(4,1)
    返回：
        result - 运行了session后的结果，运行的是Y = WX + b

    """

    np.random.seed(1)  # 指定随机种子

    X = np.random.randn(3, 1)
    W = np.random.randn(4, 3)
    b = np.random.randn(4, 1)

    Y = tf.add(tf.matmul(W, X), b)  # tf.matmul是矩阵乘法
    # Y = tf.matmul(W,X) + b #也可以以写成这样子

    # 创建一个session并运行它
    sess = tf.compat.v1.Session()  # 这里也是一样的原因和上文
    result = sess.run(Y)

    # session使用完毕，关闭它
    sess.close()

    return result

def sigmoid(z):
    """
    实现使用sigmoid函数计算z

    参数：
        z - 输入的值，标量或矢量

    返回：
        result - 用sigmoid计算z的值

    """

    # 创建一个占位符x，名字叫“x”
    x = tf.compat.v1.placeholder(tf.float32, name="x")

    # 计算sigmoid(z)
    sigmoid1 = tf.sigmoid(x)

    # 创建一个会话，使用方法二
    with tf.compat.v1.Session() as sess:
        result = sess.run(sigmoid1, feed_dict={x: z})

    return result


def one_hot_matrix(lables, C):
    """
    创建一个矩阵，其中第i行对应第i个类号，第j列对应第j个训练样本
    所以如果第j个样本对应着第i个标签，那么entry (i,j)将会是1

    参数：
        lables - 标签向量
        C - 分类数

    返回：
        one_hot - 独热矩阵

    """

    # 创建一个tf.constant，赋值为C，名字叫C
    C = tf.constant(C, name="C")

    # 使用tf.one_hot，注意一下axis
    one_hot_matrix = tf.one_hot(indices=lables, depth=C, axis=0)

    # 创建一个session
    sess = tf.compat.v1.Session()

    # 运行session
    one_hot = sess.run(one_hot_matrix)

    # 关闭session
    sess.close()

    return one_hot


def ones(shape):
    """
    创建一个维度为shape的变量，其值全为1

    参数：
        shape - 你要创建的数组的维度

    返回：
        ones - 只包含1的数组
    """

    # 使用tf.ones()
    ones = tf.ones(shape)

    # 创建会话
    sess = tf.compat.v1.Session()
    # 运行会话
    ones = sess.run(ones)

    # 关闭会话
    sess.close()

    return ones


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    '''保证sess.run()能够正常运行'''

    #print("sigmoid(0) = " + str(sigmoid(0)))
    #print("sigmoid(12) = " + str(sigmoid(12)))
    '''
    labels = np.array([1, 2, 3, 0, 2, 1])
    one_hot = one_hot_matrix(labels, C=4)
    print(str(one_hot))
    实现独热的代码
    '''
    '''
    print("ones = " + str(ones([3,4,5])))
    实现输出给定的全一或全0矩阵
    '''
    