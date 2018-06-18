# !/usr/bin/python
# -*- coding: utf-8 -*-


from numpy import *
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


# 测试数据集
def loadDataSet():
    """
    :return: 第一个变量是进行词条切分后的文档集合, 第二个变量是类别标签
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字, 0代表正常言论
    return postingList, classVec


# 创建词汇列表
def createVocabList(dataSet):
    """
    :return: 不重复的词表
    """
    vocabSet = set([])  # 创建一个空集
    for document in dataSet:  # 遍历数据集
        vocabSet = vocabSet | set(document)  # 创建两个集合的合并
    return list(vocabSet)


# 去重之后的词表
def setOfWords2Vec(vocabList, inputSet):
    """
    词频向量
    :param vocabList: 词表
    :param inputSet: 测试集
    :return: 去重之后的词表
    """
    returnVec = [0] * len(vocabList)  # 创建 跟 词表 长度等价 的 都是0的向量
    for word in inputSet:  # 遍历测试集
        if word in vocabList:  # 如果 测试集当前的单词 出现 在词汇列表里
            returnVec[vocabList.index(word)] = 1  # 把 词汇列表的当前单词的位置 在 向量 同样地方置为1
        else:
            print u"当前单词: %s 没有在我的字典里!" % word
    return returnVec


# 朴素贝叶斯分类器 训练函数
# 返回每个字出现的概率, 以及类别标签出现的概率
def trainNB0(trainMatrix, trainCategory):
    """
    :param trainMatrix: 去重之后的出现侮辱性文字的向量
    :param trainCategory: 类别标签
    :return: P0(w|c)  P1(w|c)    辱骂文档的概率
    """
    numTrainDocs = len(trainMatrix)  # 计算向量长度  6
    numWords = len(trainMatrix[0])  # 计算一个向量的元素长度  32
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算 辱骂类别 在  总类别   中出现的次数
    """这个函数第一个缺陷是:
    要计算多个概率的乘积以获得文档属于某个类别的概率, 即计算p(W0|1)p(W1|1), 如果其中一个概率为0, 那么乘积
    最后也是0, 为降低这种概率, 将所有词的出现数初始化为1, 并将分母初始化为2
    p0Num = zeros(numWords)     # 原先
    p1Num = zeros(numWords)     # 原先
    p0Denom = 0.0  # P1分母       # 原先
    p1Denom = 0.0  # P2分母       # 原先"""
    p0Num = ones(numWords)  # 初始化 非辱骂性文字 向量(分子)
    p1Num = ones(numWords)  # 初始化 辱骂性文字 向量(分子)
    p0Denom = 2.0  # P1分母
    p1Denom = 2.0  # P2分母
    for i in range(numTrainDocs):  # 遍历向量 6
        if trainCategory[i] == 1:  # 如果当前类别标签是 辱骂性文字
            p1Num += trainMatrix[i]  # 遍历每个辱骂性文档, 有辱骂的词就在其词表上+1
            p1Denom += sum(trainMatrix[i])  # 计算当前词表的辱骂出现的次数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    """另一个问题是下溢出, 由于太多很小的数想成造成的
    当计算乘积p(W0|Ci)p(W1|Ci)时, 由于大部分因子都很小, 所以程序会下溢出或者得到不正确的答案, 最后会四舍五入
    得到0, 解决办法是对乘积取自然对数, 在代数中ln(a*b)=ln(a)+ln(b)
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    """
    p1Vect = log(p1Num / p1Denom)  # 矩阵中每个辱骂词的出现的次数 / 总共出现的次数
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类器
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    :param vec2Classify: 训练集
    :param p0Vec: 出现侮辱性文字概率的向量
    :param p1Vec: 出现非侮辱性文字概率的向量
    :param pClass1: 出现侮辱性文字的概率
    :return: 文字类别
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()  # 初始化训练集, 类别
    myVocabList = createVocabList(listOPosts)  # 不重复的训练集
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 每个词在训练集中的出现概率
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))  # 侮辱性文字出现的概率  非侮辱性文字出现的概率  类别的概率
    # testEntry = ['love', 'my', 'dalmation']
    # thisDoc = array(setOfWords2Vec(myVocabList, testEntry))     # 算出当前训练集的每个词出现频率
    # print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)  # 属于哪个类别
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = range(50);
    testSet = []  # create test set
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    print 'the error rate is: ', float(errorCount) / len(testSet)
    # return vocabList,fullText


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = [];
    classList = [];
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen);
    testSet = []  # create test set
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ', float(errorCount) / len(testSet)
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = [];
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0]


if __name__ == '__main__':
    listOposts, listClasses = loadDataSet()  # 创建测试集, 类别标签
    myVocabList = createVocabList(listOposts)  # 创建不重复的测试集
    """['cute', 'love', 'help', 'garbage', 'quit', 'I', 'problems', 'is', 'park', 'stop', 'flea', 
    'dalmation', 'licks', 'food', 'not', 'him', 'buying', 'posting', 'has', 'worthless', 'ate', 'to', 
    'maybe', 'please', 'dog', 'how', 'stupid', 'so', 'take', 'mr', 'steak', 'my'"""
    result = setOfWords2Vec(myVocabList, listOposts[0])  # 测试集去重之后的词表
    """[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]"""
    trainMat = []
    for postinDoc in listOposts:  # 遍历每个测试集
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 非辱骂性文字的概率   每个辱骂性文字出现的概率  另一个概率
    p0V, p1V, pAbusive = trainNB0(trainMat, listClasses)
    """[-3.04452244 -3.04452244 -3.04452244 -2.35137526 -2.35137526 -3.04452244
 -3.04452244 -3.04452244 -2.35137526 -2.35137526 -3.04452244 -3.04452244
 -3.04452244 -2.35137526 -2.35137526 -2.35137526 -2.35137526 -2.35137526
 -3.04452244 -1.94591015 -3.04452244 -2.35137526 -2.35137526 -3.04452244
 -1.94591015 -3.04452244 -1.65822808 -3.04452244 -2.35137526 -3.04452244
 -3.04452244 -3.04452244]
 
    0.5"""
    testingNB()