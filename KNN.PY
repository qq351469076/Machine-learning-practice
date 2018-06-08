# !/usr/bin/python
# encoding: utf-8


import sys
import numpy as np
import operator
import collections

reload(sys)
sys.setdefaultencoding('utf-8')


# 训练集
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# kNN算法实现之一
def classify0(inX, dataSet, labels, k):
    """
    :param inX: 要测试的数据
    :param dataSet: 用于训练的数据(训练集)
    :param labels: 训练集各样本对应的标签
    :param k: 用于选择最近邻居的数目
    :return: 返回发生频率最高的元素标签
    """
    # 使用欧氏距离公式, 计算两个向量点xA和xB之间的距离
    dataSetSize = dataSet.shape[0]  # 查看数组的第一个维度, 列数
    # (a1 - b1) (a2 - b2) (a3 - b3)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 用将测试样本变成和训练集一样大的矩阵, 计算两矩阵元素级别上的差
    sqDiffMat = diffMat ** 2  # 结果平方
    sqDistances = sqDiffMat.sum(axis=1)  # 行结果相加, 算出距离
    distances = sqDistances ** 0.5  # 开方
    sortedDistIndicies = distances.argsort()  # 从小到大排序后返回其索引 [2 3 1 0]
    # 计算完所有点之间的距离后, 可以对数据按照从小到大的次序排序, 然后确定前k个距离最小元素所在的主要分类, k总是正整数
    classCount = {}  # 创建一个字典,存储标签和出现次数
    for i in range(k):  # 选择距离最小的k个点
        voteIlabel = labels[sortedDistIndicies[i]]  # 查找符合样本的标签类型
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 若字典中存在该标签，则直接加1；若不存在，则先初始化为0，再加1
        '''
        第1次处理结束时：classCount = {'B': 1}
        第k次处理结束时：classCount = {'A': 1, 'B': 2}
        '''
    # 倒排并返回出现次数最多的标签类型
    '''sortedClassCount = [('B', 2), ('A', 1)]'''
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# kNN实现算法之二, 列表推导式
def classify1(inX, dataSet, Labels, k):
    # 计算距离
    dist = np.sum((inX - dataSet) ** 2, axis=1) ** 0.5  # 利用numpy中的broadcasting
    # k个最近的标签
    k_labels = [Labels[index] for index in dist.argsort()[0: k]]
    # 出现次数最多的标签即为最终类别
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label


def gaussian(dist, sigma=10.0):
    weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return weight


# kNN实现算法之三, 算法优化
def classify2(inX, dataSet, Labels, k):
    dataSetSize = dataSet.shape[0]  # 获取数组行数
    # 计算欧几里得距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # xA0-xB0,将inX变为等同于dataSet的矩阵
    sqDiffMat = diffMat ** 2  # (xA0-xB0)^
    sqDistances = sqDiffMat.sum(axis=1)  # (xA0-xB0)^+(xA1-xB1)^,对2维求和，变为1维数组
    distances = sqDistances ** 0.5  # √((xA0-xB0)^+(xA1-xB1)^)
    sortedDistIndicies = distances.argsort()  # 增序排列
    # 算法优化, 高斯衰减优化, 权重
    weightCount = {}
    for i in range(k):
        weight = gaussian(distances[sortedDistIndicies[i]])
        weightCount[labels[sortedDistIndicies[i]]] = weightCount.get(labels[sortedDistIndicies[i]], 0) + weight
    sortedWeightCount = sorted(weightCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedWeightCount[0][0]


if __name__ == '__main__':
    group, labels = createDataSet()
    # result = classify0([0, 0], group, labels, 3)
    # result = classify1([0, 0], group, labels, 3)
    result = classify2([0, 0], group, labels, 3)
    print(result)
