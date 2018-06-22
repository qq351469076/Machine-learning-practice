# !/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import operator
from math import log

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


# 程序清单3-1 计算数据集的信息增益
def calcshannonent(dataset):
    """
    :return: 训练集的信息增益, 分类越多, 熵越高, 信息增益值越低
    """
    numentries = len(dataset)  # 计算样本数量
    labelcounts = {}  # 定义类别
    for featvec in dataset:
        currentlabel = featvec[-1]  # 获取类别
        if currentlabel not in labelcounts.keys():  # 如果该类别是第一次出现
            labelcounts[currentlabel] = 0  # 初始化该类别的值为0
        labelcounts[currentlabel] += 1  # 该类别出现的次数+1
    shannonent = 0.0  # 定义原始香农熵
    for key in labelcounts:  # 遍历类别
        prob = float(labelcounts[key]) / numentries  # 该类别出现的概率     P(xi) = 该分类出现的次数/总分类
        shannonent -= prob * log(prob, 2)  # 计算信息增益值         H(信息增益) = 概率*log2(概率)
    return shannonent


# 程序清单3-2 按照给定特征划分数据集
def splitdataset(dataset, axis, value):
    """
    :param axis: 要划分的特征标签索引
    :param value: 样本特征具体值
    :return: 符合条件的数据子集, 但不包括传进来的那列特征!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    retdataset = []
    for featvec in dataset:
        if featvec[axis] == value:  # 如果该样本的当前特征值是传进来的值
            reducedfeatvec = featvec[:axis]  # 截取从0到特征标签索引前面的特征
            reducedfeatvec.extend(featvec[axis + 1:])  # 截取并且追加 从特征标签索引后面+1的位置  到后面全部的特征
            retdataset.append(reducedfeatvec)
    return retdataset


# 程序清单3-3 选择最好的数据集划分方式
def choosebestfeaturetosplit(dataset):
    """
    :return: 最佳特征标签索引
    """
    numfeatures = len(dataset[0]) - 1  # 计算特征长度, 去掉类别
    baseentropy = calcshannonent(dataset)
    bestinfogain, bestfeature = 0.0, -1  # 最佳的信息增益 最佳特征标签索引值
    for i in range(numfeatures):  # 遍历特征
        featlist = [example[i] for example in dataset]  # 生成 特征 列 所有特征值  列表
        uniquevals = set(featlist)  # 去重特征值
        newentropy = 0.0  # 初始化熵
        for value in uniquevals:  # 遍历特征值
            subdataset = splitdataset(dataset, i, value)  # 每次划分出不同的特征不同的特征值, 没有一次是重复的
            prob = len(subdataset) / float(len(dataset))  # 计算当前数据子集的概率
            newentropy += prob * calcshannonent(subdataset)  # 累加当前特征的熵
        infogain = baseentropy - newentropy  # 获得当前特征的信息增益
        if infogain > bestinfogain:  # 如果当前的数据增益比记录
            bestinfogain = infogain  # 覆盖之前的
            bestfeature = i  # 最佳特征索引值
    return bestfeature


# 多数表决类别
def majoritycnt(classlist):
    """
    :param classlist: 类别
    :return: 出现次数最多的类别
    """
    classcount = {}
    for vote in classlist:
        if vote not in classcount.keys():
            classcount[vote] = 0
        classcount[vote] += 1
    sortedclasscount = sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]


# 程序清单3-4 创建树
def createtree(dataset, labels):
    classlist = [example[-1] for example in dataset]  # 获取所有类别
    if classlist.count(classlist[0]) == len(classlist):  # 递归基线条件之一, 到达叶子结点并且所有类别签相同
        return classlist[0]
    if len(dataset[0]) == 1:  # 递归基线条件之二, 消耗完所有特征, 类别仍然不是唯一, 多数表决类别
        return majoritycnt(classlist)
    bestfeat = choosebestfeaturetosplit(dataset)
    bestfeatlabel = labels[bestfeat]  # 获取最佳的数据特征标签值
    mytree = {bestfeatlabel: {}}  # 创建字典, key为最佳特征标签值, value为空字典
    del (labels[bestfeat])  # 删除原先labels里面最佳特征索引值元素
    featvalues = [example[bestfeat] for example in dataset]  # 生成所有样本的最佳特征的列
    uniquevals = set(featvalues)
    for value in uniquevals:
        sublabels = labels[:]  # 获取剩下的特征标签
        # 递归自己, 按照给定特征划分数据集(给定数据集, 最佳特征标签索引值, 不重复的样本特征), 除去之前最佳特征之后的所有标签
        mytree[bestfeatlabel][value] = createtree(splitdataset(dataset, bestfeat, value), sublabels)
        # print(createtree(splitdataset(dataset, bestfeat, value), sublabels))
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
