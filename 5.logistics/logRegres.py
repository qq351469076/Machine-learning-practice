# !/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')


# 训练集
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# Sigmoid函数, 得到一个范围在0~1之前的值
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


# 梯度上升算法
def gradAscent(dataMatIn, classLabels):
    """
    :param dataMatIn: 100 x 3 训练集
    :param classLabels: 目标值
    :return: 回归系数
    """
    dataMatrix = np.mat(dataMatIn)  # 转换成矩阵
    labelMat = np.mat(classLabels).transpose()  # 转换成矩阵, 在转置(100,1)
    m, n = np.shape(dataMatrix)  # 查看维度(100,3)
    alpha = 0.001  # 向目标移动的步长
    maxCycles = 500  # 迭代次数
    weights = np.ones((n, 1))   # 回归系数
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 矩阵乘积, sigmoid函数计算
        error = (labelMat - h)  # 向量减法, 计算真实类别与预测类别的差值
        weights = weights + alpha * dataMatrix.transpose() * error  # 矩阵乘积, 按照该差值的方向调整回归系数
    return weights


# 画出最佳拟合直线
def plotBestFit(weights):
    """
    :param weights: 回归系数
    """
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    # weights = weights.getA()    # martix转换ndarray数组
    n = np.shape(dataArr)[0]  # 获取样本数量
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):  # 遍历所有样本
        if int(labelMat[i]) == 1:  # 类别为1
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:  # 类别为0
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    :param dataMatrix: 训练集
    :param classLabels: 目标值
    :param numIter: 迭代次数
    :return: 回归系数
    """
    m, n = np.shape(dataMatrix)  # (3, 2)
    weights = np.ones(n)  # [1, 1, 1]
    for j in range(numIter):    # 迭代150次
        dataIndex = range(m)    # 生成样本索引, [0, 1, 2]
        for i in range(m):  # 遍历所有样本
            alpha = 4 / (1.0 + j + i) + 0.0001  # alpha随迭代而减少
            randIndex = int(np.random.uniform(0, len(dataIndex)))  # 随机生成一个数字, 范围0-3, 因为这个常数而趋于0
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            print('alone', dataMatrix[randIndex], weights)
            print('*', dataMatrix[randIndex] * weights)
            print('sum', sum(dataMatrix[randIndex] * weights))
            print(h)
            break
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
        break
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt');
    frTest = open('horseColicTest.txt')
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate


def multiTest():
    numTests = 10;
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))


if __name__ == '__main__':
    dataset, label = loadDataSet()
    # weights = gradAscent(dataset, label)
    # plotBestFit(weights)
    # weights = stocGradAscent0(np.array(dataset), label)
    # plotBestFit(weights)
    stocGradAscent1(dataset, label)
