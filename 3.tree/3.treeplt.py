# !/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import matplotlib.pyplot as plt

reload(sys)
sys.setdefaultencoding('utf-8')

decisionnode = dict(boxstyle='sawtooth', fc='0.8')  # 决策节点样式
leafnode = dict(boxstyle='round4', fc='0.8')  # 叶子结点样式
arrow_args = dict(arrowstyle='<-')  # 箭头样式


# 绘制带箭头的注解
def plotnode(nodetxt, centerpt, parentpt, nodetype):
    """
    :param nodetxt: 箭头注释
    :param centerpt: 箭头终点
    :param parentpt: 箭头起点
    :param nodetype: 节点的类型
    :return:
    """
    createplot.axl.annotate(nodetxt, xy=parentpt, xycoords='axes fraction', xytext=centerpt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodetype, arrowprops=arrow_args)


# 创建绘图区域
def createplot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()  # 清空绘图区
    createplot.axl = plt.subplot(111, frameon=False)
    plotnode('decision node', (0.5, 0.1), (0.1, 0.5), decisionnode)
    plotnode('leaf node', (0.8, 0.1), (0.3, 0.8), leafnode)
    plt.show()


# 获取叶子节点的数目
def getnumleafs(mytree):
    numleafs = 0
    firststr = mytree.keys()[0]
    seconddict = mytree[firststr]
    for key in seconddict.keys():
        if type(key).__name__ == 'dict':
            numleafs += getnumleafs(seconddict[key])
        else:
            numleafs += 1
    return numleafs


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


if __name__ == '__main__':
    # createplot()
    mytree = retrieveTree(0)
    leafnums = getnumleafs(mytree)
    print(leafnums)
