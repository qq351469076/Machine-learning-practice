# !/usr/bin/python
# encoding: utf-8
import sys
import numpy as np
import operator
import collections
import matplotlib.pyplot as plt
from os import listdir

reload(sys)
sys.setdefaultencoding('utf-8')


# 2.1.1 模拟训练集
def createDataSet():
    """
    :return: 样本集 类别
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 程序清单2-2 解析本地文本到向量的转换
def file2matrix(filename):
    """
    :param filename: 训练集文件名
    :return: 训练集向量, 训练集对应的类别
    """
    with open(filename, 'r') as file:
        arrayLines = file.readlines()  # 读取文件中所有数据
        numberOfLines = len(arrayLines)  # 计算有多少样本
        returnMat = np.zeros((numberOfLines, 3))  # 创建等长numberOfLines, 3列的向量
        classLabelVector = []  # 用来存放类别的列表
        index = 0  # 定位向量里的位置
        for line in arrayLines:  # 遍历每条样本
            line = line.strip()  # 去除两边空格
            listFromLine = line.split('\t')  # 切割中间的制表位, 变成一个列表  ['40920', '8.326976', '0.953952', '3']
            returnMat[index, :] = listFromLine[0:3]  # 当前样本前三个元素赋值到当前向量index位置里面去
            classLabelVector.append(int(listFromLine[-1]))  # 抽出当前样本的类别放到类别里面去
            index += 1
    return returnMat, classLabelVector


# 2.2.2 使用Matplotlib创建约会散点图(优化后)
def drawScatter1():
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    fig = plt.figure()  # 创建matlib实例
    ax = fig.add_subplot(111)  # 创建子画板区域
    # 使用第1个和第2个特征(每年飞行里程数 & 玩视频游戏的时间), 第一个参数代表x轴, 加上颜色区分像素点的不同
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * np.array(datingLabels),
               15.0 * np.array(datingLabels))
    plt.show()  # 绘制


# 程序清单2-3 归一化特征值
def autoNorm(dataSet):
    """
    :param dataSet: 需要进行归一化的训练集
    :return: 归一化之后的训练集  差值的数据范围  最小值
    """
    minValues = dataSet.min(0)  # 获得训练集中每列的最小值
    maxValues = dataSet.max(0)  # 获得最大值
    ranges = maxValues - minValues  # 最大值减去最小值求差，得数据的范围
    m = dataSet.shape[0]  # 获得训练集里的样本数量
    normDataSet = dataSet - np.tile(minValues, (m, 1))  # X - min, 每个特征减去当前特征的最小值
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # X - min / Xmax - Xmin, 上面运算结果在除以数据范围, 默认区间在[0, 1]
    return normDataSet, ranges, minValues


# 程序清单2-4 针对约会网站的分类器测试
def dataClassTest():
    testRatio = 0.10  # 训练集的10%数据拿来测试
    dataSet, dataLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minValues = autoNorm(dataSet)
    m = normMat.shape[0]
    numTestVecs = int(m * testRatio)  # 确定测试集的数量
    errorCount = 0.0  # 错误计数器
    for i in range(numTestVecs):
        # 传入一条样本、全部训练集、全部类别、k值, 获得分类的结果
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], dataLabels[numTestVecs:m], 3)
        print(u"分类的结果为: %d, 正确类别为: %d" % (classifierResult, dataLabels[i]))
        if classifierResult != dataLabels[i]:  # 错误计数+1
            errorCount += 1
    print(u"错误率: %f" % (errorCount / float(numTestVecs)))


# 程序清单2-5 约会网站预测
def classifyPerson():
    resultList = [u'一点也不喜欢', u'一般般吧', u'非常喜欢']  # 定义类别
    ffMiles = float(raw_input(u"每年飞行里程数?"))
    percentTats = float(raw_input(u"每年在视频游戏消耗的百分比"))
    iceCream = float(raw_input(u"每年消耗的冰激凌公升数"))
    dataArray, dataLabels = file2matrix("datingTestSet.txt")
    normMat, ranges, minVals = autoNorm(dataArray)
    inArray = np.array([ffMiles, percentTats, iceCream])  # 把输入的信息转换成样本
    normInArray = (inArray - minVals) / ranges  # 对样本进行归一化处理
    classifyResult = classify0(normInArray, normMat, dataLabels, 3)
    print(u"你对这个人的印象是: ", resultList[classifyResult - 1])


# 2.3.1 将本地文字图片转换为向量
def img2vector(filename):
    """
    :param filename: 本地文字图片名字
    :return: 文字图片向量
    """
    returnVector = np.zeros((1, 1024))
    with open(filename, 'r') as file:
        for i in range(32):  # 遍历32像素高度
            lineStr = file.readline()  # 读取每行
            for j in range(32):  # 遍历32像素宽度
                returnVector[0, 32 * i + j] = int(lineStr[j])  # 将该行上第j个数据存进数组第i行第j列中
    return returnVector


# 程序清单2-6 手写数字识别系统测试代码
def handwritingClassTest():
    hwLabels = []  # 类别

    trainingFileList = listdir("./trainingDigits")  # 遍历训练集目录下所有文件
    m = len(trainingFileList)  # 获取训练集目录下的样本数量
    trainingMat = np.zeros((m, 1024))  # 初始化训练集矩阵
    for i in range(m):  # 遍历所有训练集下的样本
        fileNameStr = trainingFileList[i]  # 获取文件名全称，如 3_107.txt
        fileStr = fileNameStr.split('.')[0]  # 根据 . 划分，获取文件名 如 3_107
        classNum = int(fileStr.split('_')[0])  # 根据 _ 划分，获取该文件表示的真实数字 如 3
        hwLabels.append(classNum)  # 将该数字标签放入类别
        trainingMat[i, :] = img2vector('./trainingDigits/%s' % fileNameStr)  # 将文字图片向量赋值到当前的向量位置中

    testFileList = listdir("./testDigits")
    errorCount = 0.0    # 错误统计
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("./testDigits/%s" % fileNameStr)
        classifyResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)   # 估计错误率
        print(u"当前分类: %d, 正确结果是: %d" % (classifyResult, classNum))
        if classifyResult != classNum:
            errorCount += 1.0
    print(u"总共出错了: %d 次" % errorCount)
    print(u"错误率: %f" % (errorCount / float(mTest)))


# 程序清单2-1 k近邻算法
def classify0(inX, dataSet, labels, k):
    """
    :param inX: 一条样本
    :param dataSet: 训练集
    :param labels: 类别
    :param k: 用于选择最近邻居的数目
    :return: 返回训练集里 距离  样本  最近的那个样本对应的类别
    """
    dataSetSize = dataSet.shape[0]  # 获得训练集样本数量
    # (xA0 - xB0) (xA1 - xB1) (xA2 - xB2)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 将样本转换成同训练集同等规模的矩阵, 计算两矩阵元素级别上的差
    sqDiffMat = diffMat ** 2  # 上个运算的结果按照每个元素独立平方, 最后整体运算
    sqDistances = sqDiffMat.sum(axis=1)  # 所有特征值按行累加
    distances = sqDistances ** 0.5  # 上个运算结果最后开方, 算出样本到训练集的所有距离
    sortedDistIndicies = distances.argsort()  # 从小到大排序后返回其索引 [2 3 1 0]  下标2最小, 下标0最大
    classCount = {}  # 存储训练集类别和其出现次数
    for i in range(k):  # 自定义邻居数量
        voteIlabel = labels[sortedDistIndicies[i]]  # 查找符合样本的标签类型
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 若字典中存在该标签，则直接加1；若不存在，则先初始化为0，再加1
        '''
        第1次处理结束时：classCount = {'B': 1}
        第k次处理结束时：classCount = {'A': 1, 'B': 2}
        '''
    # 按照每个元素的下标1进行倒排并返回出现次数最多的标签类型
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
    # group, labels = createDataSet()   # 创建测试集
    # data, label = file2matrix('datingTestSet.txt')  # 创建训练集
    # print(data)
    # print(label[:20])
    # drawScatter1()
    # dataSet, ranges, min = autoNorm(data)  # [0.44832535 0.39805139 0.56233353]
    # print(dataSet)
    # print(ranges)
    # print(min)
    # dataClassTest()
    # classifyPerson()
    # testSet = img2vector('./testDigits/0_0.txt')
    handwritingClassTest()
    # result = classify0([0, 0], group, labels, 3)
    # print(result)
    # result = classify1([0, 0], group, labels, 3)
    # result = classify2([0, 0], group, labels, 3)
    # print(result)
