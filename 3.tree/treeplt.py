# !/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import matplotlib.pyplot as plt
import matplotlib
from tree import createdataset

reload(sys)
sys.setdefaultencoding('utf-8')

# -------------------------------------------定义文本框和箭头格式-------------------------------------------------------
myfont = matplotlib.font_manager.FontProperties(fname='C:/Windows/Fonts/simkai.ttf')  # 支持中文字体
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 决策节点样式
leafNode = dict(boxstyle="round4", fc="0.8")  # 叶子结点杨慧
arrow_args = dict(arrowstyle="<-")  # 箭头样式


# --------------------------------------------------------------------------------------------------------------------
# 绘制箭头
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    :param nodeTxt: 文字
    :param centerPt: 箭头终点
    :param parentPt: 箭头起点
    :param nodeType: 箭头类型
    :return:
    """
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, fontproperties=myfont)


# 创建绘画区
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')  # 创建绘图面板
    fig.clf()  # 清空绘画区
    axprops = dict(xticks=[], yticks=[])  # 包含了x,y坐标的字典
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 创建子面板
    plotTree.totalW = float(getNumLeafs(inTree))  # 计算决策树宽度     3.0
    plotTree.totalD = float(getTreeDepth(inTree))  # 计算决策树高度    2.0
    plotTree.xOff = -0.5 / plotTree.totalW  # 计算x偏移量         -0.16666666666666666
    plotTree.yOff = 1.0  # 计算y偏移量
    plotTree(inTree, (0.5, 1.0), '')  # 传值, 父节点坐标
    plt.show()


# 获取叶子节点, 节点只有唯一类别的被称之为叶子结点
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获取树的高度, 第一关键词之后开始计数, 也包括叶子结点
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


# 在父节点间填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    """
    :param cntrPt: (0.5. 1.0)
    :param parentPt: 父节点坐标  (0.5, 1.0)
    :param txtString: 父节点文字
    :return:
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 0.25
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]  # 1.0
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


# 绘制树形图
def plotTree(myTree, parentPt, nodeTxt):
    """
    :param myTree: 决策树
    :param parentPt: 父节点坐标(0.5, 1.0)
    :param nodeTxt: 父节点的文本信息
    :return:
    """
    numLeafs = getNumLeafs(myTree)  # 获取树宽度     3
    depth = getTreeDepth(myTree)  # 获取树高度       2
    firstStr = myTree.keys()[0]  # 获取第一个关键词     no surfacing
    # plotTree.totalW为传进来的宽度, plotTree.totalD为传进来的高度
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # (0.5, 1.0)
    plotMidText(cntrPt, parentPt, nodeTxt)  # 创建父节点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制第一个决策节点
    secondDict = myTree[firstStr]  # 获取第二个关键词
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # 计算接下来的偏移量
    for key in secondDict.keys():  # 遍历第二个关键词的所有key
        if type(secondDict[key]).__name__ == 'dict':  # 如果key的类型是字典, 创建父节点
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW  # 计算偏移量   0.6666666666666666
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)  # 绘制叶子结点
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 测试用的树
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


# 决策树分类器
def classify(inputTree, featLabels, testVec):
    """
    创建树会返回树的字典结构, 但是第一个关键词, 计算器不知道在特征标签的哪个位置, 于是需要此方法来查找第一关键词在
    列表中的索引
    :param inputTree:   决策树模型
    :param featLabels:  特征标签
    :param testVec: 测试样本
    :return:
    """
    firststr = inputTree.keys()[0]
    seconddict = inputTree[firststr]
    featIndex = featLabels.index(firststr)
    for key in seconddict.keys():
        if testVec[featIndex] == key:
            if type(seconddict[key]).__name__ == 'dict':
                classLabel = classify(seconddict[key], featLabels, testVec)
            else:
                classLabel = seconddict[key]
    return classLabel


# 决策树储存
def storeTree(mytree, filename):
    """
    构造决策树很耗时, 为了节省时间, 每次执行分类调用已经构造好的决策树
    """
    import pickle
    with open(filename, 'w') as f:
        pickle.dump(mytree, f)


# 决策树读取
def grabTree(filename):
    import pickle
    with open(filename, 'r') as f:
        return pickle.load(f)


if __name__ == '__main__':
    mytree = retrieveTree(0)
    # createPlot(mytree)
    # print mytree
    # leafnums = getNumLeafs(mytree)
    # print(leafnums)
    # maxdepth = getTreeDepth(mytree)
    # print maxdepth
    dataset, labels = createdataset()
    result = classify(mytree, labels, [1, 0])
    print(result)
    # storeTree(mytree, 'classifierStorage.txt')
    # result = grabTree('classifierStorage.txt')
    # print(result)
