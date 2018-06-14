# !/usr/bin/python
# encoding: utf-8


import sys
import numpy as np
import operator
import collections
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

reload(sys)
sys.setdefaultencoding('utf-8')


# 训练集
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 将文本转换成向量矩阵
def file2matrix(filename):
    """
    返回样本数据数组和标签
    """
    with open(filename, 'r') as file:
        arrayLines = file.readlines()  # 读取文件中所有行数据
        numberOfLines = len(arrayLines)  # 文件行数
        returnMat = np.zeros((numberOfLines, 3))  # 创建 文件行数的数量的行, 3列都是0的矩阵
        classLabelVector = []  # 创建标签值列表
        index = 0
        for line in arrayLines:  # 遍历行
            line = line.strip()  # 去除两边空格
            listFromLine = line.split('\t')  # 切割所有行变成列表形式
            returnMat[index, :] = listFromLine[0:3]  # 获取该行的前3个数据(飞行历程, 游戏时间, 冰激凌数), 把类别单独抽出来
            classLabelVector.append(int(listFromLine[-1]))  # 获取该样本数据的类别, 抽成一个标签值列表
            index += 1
    return returnMat, classLabelVector


# 使用Matplotlib(2.2 示例：使用k-近邻算法改进约会网站的配对效果)
def drawScatter1():
    # 获取矩阵和标签
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 使用第2个和第3个特征(玩视频游戏所耗时间百分比 & 每周消费的冰淇淋公升数)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * np.array(datingLabels),
               15.0 * np.array(datingLabels))
    plt.show()


# 归一化
def autoNorm(dataSet):
    minValues = dataSet.min(0)  # 返回数据集中每列上的最小值
    maxValues = dataSet.max(0)  # 每列上的最大值
    ranges = maxValues - minValues  # 求差，得数据的范围
    normDataSet = np.zeros(shape=np.shape(dataSet))  # 根据shape创建都为0的矩阵
    m = dataSet.shape[0]  # 样本数量
    normDataSet = dataSet - np.tile(minValues, (m, 1))  #
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # (X - min) / (Xmax - Xmin), 区间在[0, 1]
    return normDataSet, ranges, minValues


# k近邻分类器
def dataClassTest():
    testRatio = 0.10  # 训练集的10%数据拿来测试
    dataSet, dataLabels = file2matrix("datingTestSet.txt")  # 读取数据
    normMat, ranges, minValues = autoNorm(dataSet)  # 归一化处理
    m = normMat.shape[0]  # 样本数量
    numTestVecs = int(m * testRatio)  # 确定测试样本的数量
    errorCount = 0.0  # 错误数量统计
    for i in range(numTestVecs):  # 遍历测试集的数量，计算错误率
        # normMat[numTestVecs:m, :]表示测试向量后面的所有向量(训练向量，用于训练分类器)
        # datingLabels[numTestVecs:m]表示训练向量对应的类别(list类型)
        # 传入测试数据、训练集、训练标签、k值
        # k近邻是一种分类器
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], dataLabels[numTestVecs:m], 3)
        print("classify: %d, read answer: %d" % (classifierResult, dataLabels[i]))
        if classifierResult != dataLabels[i]:  # 判断kNN算法计算得到的分类和实际的分类是否相同, 如果预测不正确，则统计加1
            errorCount += 1
    print("total error rate is: %f" % (errorCount / float(numTestVecs)))  # 打印错误率


# 2.2.5约会网站预测信息
def classifyPerson():
    resultList = ['一点也不喜欢', '一般般吧', '非常喜欢']  # 定义三种喜欢程度，对应数据集中标签 1,2,3
    ffMiles = float(raw_input("每年飞行频率?"))  # 输入每年飞行里程数
    percentTats = float(raw_input("每年在视频游戏消耗的百分比"))  # 输入玩游戏所耗时间百分比
    iceCream = float(raw_input("每周消耗的冰激凌公升数"))  # 输入每周消费冰激凌公升数
    dataArray, dataLabels = file2matrix("datingTestSet.txt")  # 从txt中获取训练数据
    normMat, ranges, minVals = autoNorm(dataArray)  # 归一化处理
    inArray = np.array([ffMiles, percentTats, iceCream])  # 对测试数据处理，整合成数组
    normInArray = (inArray - minVals) / ranges  # 对数据做归一化处理
    classifyResult = classify0(normInArray, normMat, dataLabels, 3)  # 分类，k=3
    print("你对这个人的印象是: ", resultList[classifyResult - 1])


# 将图像转为向量
def img2vector(filename):
    returnVector = np.zeros((1, 1024))  # 初始化0数组，1行1024列
    with open(filename, 'r') as file:  # 读取文件
        for i in range(32):  # 遍历行
            lineStr = file.readline()  # 读取行
            for j in range(32):  # 遍历列
                returnVector[0, 32 * i + j] = int(lineStr[j])  # 将该行上第j个数据存进数组第i行第j列中
    return returnVector


# 手写数字识别系统测试代码
def handwritingClassTest():
    hwLabels = []  # 列表，存放训练数据集标签
    trainingFileList = listdir("./trainingDigits")  # 列出给定目录中的文件名
    m = len(trainingFileList)  # 训练样本数
    trainingMat = np.zeros((m, 1024))  # 初始化全0矩阵 m行1024列
    for i in range(m):  # 遍历训练数据
        fileNameStr = trainingFileList[i]  # 获取文件名全称，如 3_107.txt
        fileStr = fileNameStr.split('.')[0]  # 根据 . 划分，获取文件名 如 3_107
        classNum = int(fileStr.split('_')[0])  # 根据 _ 划分，获取该文件表示的真实数字 如 3
        hwLabels.append(classNum)  # 将该数字标签放入训练集标签列表中
        trainingMat[i, :] = img2vector('./trainingDigits/%s' % fileNameStr)  # 调用函数，将第i个文件内的内容转化为数组，并存储
    testFileList = listdir("./testDigits")  # 列出测试集目录中的文件名
    errorCount = 0.0  # 错误统计
    mTest = len(testFileList)  # 测试集大小
    for i in range(mTest):  # 遍历测试集
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("./testDigits/%s" % fileNameStr)
        classifyResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  # 调用函数，预测数字
        print("the classifier: %d, the real value: %d" % (classifyResult, classNum))
        if classifyResult != classNum:
            errorCount += 1.0
    print("total number of errors: %d" % errorCount)
    print("total error rate: %f" % (errorCount / float(mTest)))  # 错误率


# kNN算法书籍实现之一
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
def classify2(inX, dataSet, labels, k):
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
    # group, labels = createDataSet()
    # data, label = file2matrix('datingTestSet.txt')
    # drawScatter1()
    # dataSet, ranges, min = autoNorm(data)    # [0.44832535 0.39805139 0.56233353]
    # dataClassTest()
    # classifyPerson()
    # testSet = img2vector('./testDigits/0_0.txt')
    handwritingClassTest()
    # result = classify0([0, 0], group, labels, 3)
    # result = classify1([0, 0], group, labels, 3)
    # result = classify2([0, 0], group, labels, 3)
    # print(result)
