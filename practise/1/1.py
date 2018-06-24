# !/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import jieba  # 中文分词方法
from sklearn.feature_extraction.text import CountVectorizer  # 文本特征抽取
from sklearn.feature_extraction import DictVectorizer  # 字典特征抽取
from sklearn.feature_extraction.text import TfidfVectorizer  # 文字重要程度
from sklearn.preprocessing import MinMaxScaler  # 归一化
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.preprocessing import Imputer  # 处理缺失值
from sklearn.feature_selection import VarianceThreshold  # 特征选择
from sklearn.decomposition import PCA  # PCA降维
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯分类器
from sklearn.metrics import classification_report  # 朴素贝叶斯分类器预估
from sklearn.neighbors import KNeighborsClassifier  # k近邻估计器
from sklearn.model_selection import train_test_split  # 训练集划分测试集
from sklearn.datasets import load_iris, fetch_20newsgroups  # 鸢尾植物数据集, 新闻数据集

reload(sys)
sys.setdefaultencoding('utf-8')


# 中文分词
def cutword():
    """
    :return: 空格切分的中分分词
    """
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

    content1 = []

    [content1.append(word) for word in con1]

    return ' '.join(content1)


# 中英文分词
def countvec():
    c1 = cutword()
    cv = CountVectorizer()
    data = cv.fit_transform([c1])
    print(' '.join(cv.get_feature_names()))
    print(data.toarray())


# 文本特征抽取
def textcv():
    # vector = CountVectorizer(min_df=2, max_df=2)  # 最小出现两次, 最大也出现两次筛选出来
    vector = CountVectorizer()
    res = vector.fit_transform(["life is short,i like python"])  # 转换成词频列表
    print(vector.get_feature_names())  # 去重词表
    print(res.toarray())  # 词频向量


# 字典特征抽取
def dictvec():
    # 默认参数sparse是开启， 开启状态为sparse矩阵类型， 关闭为二维数组类型
    dict = DictVectorizer(sparse=False)  # 也可以关闭sparse, 用toarray()方式转换成二维数组
    data = dict.fit_transform([{'city': '北京', 'temperature': 100}, {'city': '深圳', 'temperature': 30}])
    print(' '.join(dict.get_feature_names()))  # 获取文本特征
    print(data)  # 词频向量


# tfidf  考量的是每个分词在文章里面的重要程度
def tfidfvec():
    c1 = cutword()
    tf = TfidfVectorizer(stop_words=["所以", "明天"])  # 过滤某些不需要的词
    data = tf.fit_transform([c1])
    print(' '.join(tf.get_feature_names()))
    print(data.toarray())


# 归一化
def minmax():
    mm = MinMaxScaler(feature_range=(2, 3))  # 可以指定归一化的区间
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)


# 标准化
def stand():
    std = StandardScaler()
    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])
    print(data)


# 处理缺失值
def im():
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    data = imputer.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
    print(data)


# 特征选择, 删除所有低方差特征
def variance():
    vt = VarianceThreshold(threshold=0.0)  # 可以设置方差范围
    data = vt.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)


# PCA数据集降维
def pca():
    pc = PCA(n_components=3)
    data = pc.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])
    print(data)


# 划分测试集
def tts():
    li = load_iris()
    print(li.data)  # 获取特征
    print(li.target)  # 获取类别
    print(li.feature_names)  # 获取特征标签
    print(li.target_names)  # 获取类别标签
    """训练集的特征值       测试集的特征值     训练集类别     测试集类别,      规模为0.25, 随机种子自定"""
    x_train, y_test, x_target, y_target = train_test_split(li.data, li.target, test_size=0.25, random_state=24)
    print(x_train, x_test)


if __name__ == "__main__":
    # dictvec()
    # textcv()
    # countvec()
    # tfidfvec()
    # minmax()
    # stand()
    # im()
    # variance()
    pca()
    # cutword()

# 总结

# 机器学习：数据， 分析获得的规律，对未知数据进行预测

# 数据来源和类型：
# 类型：离散型：区间内不可分
#        连续型：区间内可分

# scikit-learn,UCI ,kaggle

# 特征值+目标值
# 特征工程；
# 1、特征抽取：字典抽取：one-hot编码
# 文本抽取：CountVectorizer：通过词频技术，出现的不重复的单词（省略了单个字母），
# 如果是中文，要进行分词（省略了单个字）
# tfidf：重要性程度
# tf
# idf:逆文档频率log(总次数/当前文档次数)

# 2、特征预处理：某些场景需要
# 特征缩放：归一化，标准化
# 缺失值：填充，删除

# 3、特征选择：防止数据冗余，嘈杂数据
# 过滤式：删除低方差特征

# 4、降维：维度，数据都改变（运算过程不需要了解）

# 机器学习算法分类：
# 监督学习：特征值+目标值
# 分类：
# 回归：
# 标注：
# 非监督学习：特征值
# 聚类算法
