# !/usr/bin/python
# -*- coding: utf-8 -*-


import sys
from math import log

reload(sys)
sys.setdefaultencoding('utf-8')


def createdataset():
    dataset = [[1, 1, 'maybe'],
               [1, 1, 'yes'],
               # [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']
    return dataset, labels


# 根据不同的分类算出香农熵, 度量数据集的无序程度
def calcshannonent(dataset):
    numentries = len(dataset)  # 查看数据集长度
    labelcounts = {}  # 定义分类字典
    for featVec in dataset:  # 遍历数据集
        currentlabel = featVec[-1]  # 拿到当前样本的目标值
        if currentlabel not in labelcounts.keys():  # 如果当前的目标值没有在字典的key里
            labelcounts[currentlabel] = 0  # 在字典里定义新的目标值, 初始化为0
            labelcounts[currentlabel] += 1  # 该目标值的次数+1
    shannonent = 0.0  # 定义香农值
    for key in labelcounts:  # 遍历所有出现的分类
        prob = float(labelcounts[key]) / numentries  # 当前分类 / 数据集长度, 计算类别出现频率
        shannonent -= prob * log(prob, 2)  # 原始香农熵
    # 熵越高, 混合的数据越多
    return shannonent


# 按照给定特征划分数据集
def splitdataset(dataset, axix, value):
    """
    :param dataset: 要划分的数据集
    :param axix: 要划分的特征
    :param value: 特征返回值
    :return:
    """
    retdataset = []
    for featvec in dataset:
        if featvec[axix] == value:  # 如果当前样本特征符合要划分的特征
            reducedfeatvec = featvec[:axix]  # 截取从起始到要划分的特征
            reducedfeatvec.extend(featvec[axix + 1:])  # 追加当前样本特征值后面第一个值到最后, 包括目标值
            retdataset.append(reducedfeatvec)
    return retdataset


# 选择最好的数据集划分方式
def choosebestfeaturetosplit(dataset):
    numfeatures = len(dataset[0]) - 1  # 以0为起点计算单个样本特征长度
    baseentropy = calcshannonent(dataset)  # 度量数据集的无序程度, 又称原始香农熵
    bestinfogain, bestfeature = 0.0, -1  # 最佳的信息增益, 最佳特征索引值
    for i in range(numfeatures):  # 遍历单个样本特征, 同时也隐式去掉了列目标值
        featlist = [example[i] for example in dataset]  # 生成每列特征值的列表, 不包含目标值    [1, 1, 1, 0, 0]
        uniquevals = set(featlist)  # 去掉重复的列特征值     set([0, 1])
        newentropy = 0.0    # 初始化当前样本的熵
        for value in uniquevals:  # 遍历不重复的列特征值
            # 划分出特定的数据集, 筛选符合指定特征后面所有列(不包括指定特征)
            subdataset = splitdataset(dataset, i, value)    # [[1, 'no'], [1, 'no']]
            prob = len(subdataset) / float(len(dataset))  # 划分分类之后的次数 / 数据集      (计算类别出现的频率)
            newentropy += prob * calcshannonent(subdataset)  # 熵 += 出现的频率 * 数据集的无序程度
        infogain = baseentropy - newentropy  # 最佳的信息增益 - 熵    0.2421793565
        if infogain > bestinfogain:  # 如果当前的信息增益大于最佳信息增益, 赋最新值
            bestinfogain = infogain     # 最佳信息增益
            bestfeature = i     # 最佳特征索引值
    # 返回数据集中最佳索引值
    return bestfeature


if __name__ == '__main__':
    dataset, label = createdataset()
    shannonent = calcshannonent(dataset)
    print shannonent
    # retdataset = splitdataset(dataset, 0, 1)
    # print retdataset
    # bestfeature = choosebestfeaturetosplit(dataset)
    # print bestfeature
