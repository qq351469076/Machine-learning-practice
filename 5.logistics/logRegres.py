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
    每次计算整个训练集的梯度
    使用alpha * 梯度 更新回归系数的向量
    返回最佳回归系数
    :param dataMatIn: 100 x 3 训练集
    :param classLabels: 目标值
    :return: 最佳回归系数
    """
    dataMatrix = np.mat(dataMatIn)  # (100, 3)
    labelMat = np.mat(classLabels).transpose()  # (100,1)
    m, n = np.shape(dataMatrix)  # 100  3
    alpha = 0.001  # 向目标移动的步长
    maxCycles = 500  # 迭代次数
    weights = np.ones((n, 1))  # 初始化回归系数, 都是1 (3, 1)
    for k in range(maxCycles):  # 迭代500次
        h = sigmoid(dataMatrix * weights)  # 计算数据集的回归系数  (100, 1)
        error = (labelMat - h)  # 计算真实类别与预测类别的差值, (100, 1)
        weights = weights + alpha * dataMatrix.transpose() * error  # 计算梯度上升  (3, 1)
    return weights


# 画出最佳拟合直线
def plotBestFit(weights):
    """
    :param weights: 回归系数
    """
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)  # (100, 3)
    weights = weights.getA()  # martix转换ndarray数组
    n = np.shape(dataArr)[0]  # 获取样本数量
    xcord1 = []  # x1
    ycord1 = []  # x2
    xcord2 = []  # y1
    ycord2 = []  # y2
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
    x = np.arange(-3.0, 3.0, 0.1)  # 二维数组, 从-3到3, 步长0.1
    y = (-weights[0] - weights[1] * x) / weights[2]
    print(y)
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    """
    对训练集中每个样本计算梯度
    使用alpha * 梯度 更新回归系数
    返回最佳回归系数

    可以在新样本到来时对分类器进行增量更新, 因此随机梯度上升算法是一个在线学习算法, 一次性处理所有数据叫'批处理'

    :param dataMatrix: 训练集
    :param classLabels: 目标值
    :return: 最佳回归系数
    """
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)  # 初始化回归系数 (1, 1,.....1)
    for i in range(m):  # 遍历全部样本
        h = sigmoid(sum(dataMatrix[i] * weights))  # 0-1之间的数值, 计算每个样本的回归系数
        error = classLabels[i] - h  # 计算差值
        weights = weights + alpha * error * dataMatrix[i]  # 调整回归系数
    return weights


# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    alpha每次迭代都会调整, 会缓解数据波动或者高频波动, 另外永远不会减小到0, 因为里面存在一个常数项, 必须这样做的原因是为了
    保证在多次迭代以后新数据仍具有一定的影响, 每次抽取样本计算回归系数
    :param dataMatrix: 训练集
    :param classLabels: 目标值
    :param numIter: 迭代次数
    :return: 回归系数
    """
    m, n = np.shape(dataMatrix)  # (3, 2)
    weights = np.ones(n)  # [1, 1, 1]
    for j in range(numIter):  # 迭代150次
        dataIndex = range(m)  # 生成样本索引, [0, 1, 2]
        for i in range(m):  # 遍历所有样本
            alpha = 4 / (1.0 + j + i) + 0.0001  # alpha随迭代而减少
            randIndex = int(np.random.uniform(0, len(dataIndex)))  # 随机生成一个数字, 范围0-3, 因为这个常数而趋于0
            h = sigmoid(sum(dataMatrix[randIndex] * weights))  # 随机取一个样品计算回归系数
            error = classLabels[randIndex] - h  # 计算差值
            weights = weights + alpha * error * dataMatrix[randIndex]  # 计算梯度上升
            del (dataIndex[randIndex])  # 删除当前样本
    return weights


# Logistic回归分类器
def classifyVector(inX, weights):
    """
    :param inX: 训练集
    :param weights: 最佳回归系数
    :return: 类别
    """
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


# 从疝气病症预测病马的死亡率
def colicTest():
    """
    :return: 错误率
    """
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []  # 训练集
    trainingLabels = []  # 目标值
    for line in frTrain.readlines():  # 遍历训练集
        currLine = line.strip().split('\t')  # 样本数组
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))  # 添加当前样本特征值
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 1000)     # 最佳回归系数
    errorCount = 0
    numTestVec = 0.0    # 测试次数
    for line in frTest.readlines():     # 遍历测试集
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):   # 如果分类之后如果不是正确结果
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)    # 估算错误率
    print "错误率是: %f" % errorRate
    return errorRate


def multiTest():
    numTests = 10;
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))


if __name__ == '__main__':
    dataset, label = loadDataSet()
    weights = gradAscent(dataset, label)
    print(weights)
    # plotBestFit(weights)
    # weights = stocGradAscent0(np.array(dataset), label)
    # plotBestFit(weights)
    # stocGradAscent1(dataset, label)
    # colicTest()
