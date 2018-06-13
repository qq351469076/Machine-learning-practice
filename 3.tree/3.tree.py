# !/usr/bin/python
# -*- coding: utf-8 -*-


import sys
from math import log
import operator

reload(sys)
sys.setdefaultencoding('utf-8')


def createdataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


# 度量数据集的无序程度(求熵)
def calcshannonent(dataset):
    """
    无序程度 = 计算样本数量 / 计算类别数量
    :return: 数据集的无序程度
    """
    numentries = len(dataset)  # 计算样本数量
    labelcounts = {}  # 定义分类字典
    for featvec in dataset:
        currentlabel = featvec[-1]  # 当前样本的目标值
        if currentlabel not in labelcounts.keys():
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] += 1
    shannonent = 0.0  # 定义原始香农熵
    for key in labelcounts:
        prob = float(labelcounts[key]) / numentries  # 概率     P(xi) = 该分类出现的次数/总分类
        shannonent -= prob * log(prob, 2)  # 原始香农熵         H(求和) = 概率*log2(概率)
    # 分类越多, 熵越高
    return shannonent


# 按照给定特征划分数据集
def splitdataset(dataset, axis, value):
    """
    用 给定的标签索引 和 样本特征值 筛选 数据子集
    :param axis: 要划分的特征标签索引
    :param value: 样本特征具体值
    :return: 符合条件的数据子集, 但不包括传进来的那列特征!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    retdataset = []
    for featvec in dataset:
        if featvec[axis] == value:
            reducedfeatvec = featvec[:axis]
            reducedfeatvec.extend(featvec[axis + 1:])
            retdataset.append(reducedfeatvec)
    return retdataset


# 最优数据划分
def choosebestfeaturetosplit(dataset):
    """
    遍历样品的每行每列, 尽可能选择包含唯一类别的数据子集, 优先选择直接划分出分类的特征索引
    :return: 返回最佳的特征标签索引
    """
    numfeatures = len(dataset[0]) - 1  # 计算样本特征长度, 不包括目标值
    baseentropy = calcshannonent(dataset)  # 度量数据集的无序程度
    bestinfogain, bestfeature = 0.0, -1  # 初始化 最佳的信息增益 和 最佳特征标签索引值
    for i in range(numfeatures):  # 遍历每个特征
        featlist = [example[i] for example in dataset]  # 生成每列特征值的列表, 不包含目标值
        uniquevals = set(featlist)
        newentropy = 0.0  # 初始化当前样本的熵
        for value in uniquevals:  # 遍历不重复的样品特征值
            subdataset = splitdataset(dataset, i, value)  # 返回划分数据子集, 但不包括传进去的那列特征
            prob = len(subdataset) / float(len(dataset))  # 计算当前数据子集的概率
            newentropy += prob * calcshannonent(subdataset)  # 计算当前数据子集的熵
        infogain = baseentropy - newentropy  # 获得当前样本的信息增益
        if infogain > bestinfogain:  # 如果当前的数据增益比记录的大, 覆盖之前的
            bestinfogain = infogain
            bestfeature = i  # 最佳特征索引值
    return bestfeature


# 返回次数最多的分类名称
def majoritycnt(classlist):
    """
    使用最优数据划分仍不能划分出唯一类别的分组, 将挑选次数最多的类别
    :param classlist: 特征标签
    :return: 次数最多的分类名称
    """
    classcount = {}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote] = 0
        classcount[vote] += 1
    sortedclasscount = sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=true)
    return sortedclasscount[0][0]


# 创建树
def createtree(dataset, labels):
    classlist = [example[-1] for example in dataset]  # 生成所有样本的列目标值
    if classlist.count(classlist[0]) == len(classlist):  # 递归基线条件之一, 所有类标签完全相同
        return classlist[0]
    if len(dataset[0]) == 1:  # 递归基线条件之二, 使用完所有特征, 仍不能将数据集划分为包含唯一类别, 需要返回次数最多的分类
        return majoritycnt(classlist)
    bestfeat = choosebestfeaturetosplit(dataset)  # 返回最佳的数据特征标签索引值
    bestfeatlabel = labels[bestfeat]  # 获取最佳的数据特征标签值
    mytree = {bestfeatlabel: {}}  # 创建字典, key为最佳特征标签值, value为空字典
    del (labels[bestfeat])  # 删除原先labels里面最佳特征索引值元素
    featvalues = [example[bestfeat] for example in dataset]  # 生成所有样本的最佳特征的列
    uniquevals = set(featvalues)  # 去重样本特征
    for value in uniquevals:  # 遍历去重的样本特征值
        sublabels = labels[:]  # 获取剩下所有特征标签
        # 递归自己, 按照给定特征划分数据集(给定数据集, 最佳特征标签索引值, 不重复的样本特征), 除去之前最佳特征之后的所有标签
        mytree[bestfeatlabel][value] = createtree(splitdataset(dataset, bestfeat, value), sublabels)
    return mytree


if __name__ == '__main__':
    dataset, label = createdataset()
    # shannonent = calcshannonent(dataset)
    # print shannonent
    # retdataset = splitdataset(dataset, 0, 0)
    # print retdataset
    # bestfeature = choosebestfeaturetosplit(dataset)
    # print bestfeature
    mytree = createtree(dataset, label)  # {'flippers': {0: 'no', 1: {'no surfaching': {0: 'no', 1: 'yes'}}}}
    print mytree
