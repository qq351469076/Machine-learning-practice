# !/usr/bin/python
# -*- coding: utf-8 -*-


# from sklearn.model_selection import GridSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

# 数据集划分, 获取小数据集, 返回类型为字典, 可以通过各种key获取values
from sklearn.model_selection import train_test_split
def tts():
    li = load_iris()
    print(li.data)  # 获取数据集
    print(li.target)  # 获取标签值
    print(li.feature_names)  # 获取特征值
    print(li.target_names)  # 获取类别
    # 数据集的划分, 返回训练集的特征值，测试集的特征值，训练集的目标值，测试集的目标值
    # 数据集划分, 测试集规模为0.25, 随机种子自定
    x_train, x_test, y_train, y_test = train_test_split(li.data, li.target, test_size=0.25, random_state=24)
    print(x_train, x_test)


# 数据集划分, 获取大数据集, 需要下载数据, 返回类型为字典, 可以通过key获取values
from sklearn.datasets import fetch_20newsgroups
def tts1():
    # all代表下载的时候获取全部数据集, 可以自己特征处理,也可以train或test下载训练集或测试集
    # feature_names可能没有
    # 有种子可用
    news = fetch_20newsgroups(subset='all')
    print news.data


# 回归数据集
from sklearn.datasets import load_boston
def tts2():
    boston = load_boston()
    print boston.data
    print boston.target


# K-近邻算法实现入住
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
def knncls():
    # 获取数据，分析数据
    data = pd.read_csv("./train.csv")
    # 缩小数据的范围，防止运算时间过长
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")
    # 处理时间日期，分割时间，增加一些日期的详细特征
    time_value = pd.to_datetime(data['time'], unit='s')
    # 把时间格式转换成字典格式，获取年，月，日
    time_value = pd.DatetimeIndex(time_value)
    # 构造新的特征，可用的方法: weekday, day ,hour
    data['weekday'] = time_value.weekday
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    # 字典删除无用的列, 因为增加了新特征
    data = data.drop(['time'], axis=1)
    # 删除一些签到位置少的签到点, 这句话聚合place_id, 并且统计了place_id出现了多少次
    place_count = data.groupby('place_id').aggregate(np.count_nonzero)
    # 进行筛选, 这里面都是出现了三次以上的place_id
    tf = place_count[place_count.row_id > 3].reset_index()
    # 保留出现三次以上的place_id, 重新进行赋值
    data = data[data['place_id'].isin(tf.place_id)]
    # 取出特征值和目标值
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)
    # 进行数据的分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 进行标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    # 上面经过fit_transform处理特征处理, 如果在对测试集进行特征处理, 两个数据集的大小不一样, 结果会发生变动
    # 统一取值
    x_test = std.transform(x_test)
    # estimaotr估计器流程
    knn = KNeighborsClassifier()
    # fit数据
    knn.fit(x_train, y_train)
    # 预测结果
    # 得出准确率
    score = knn.score(x_test, y_test)
    print score
    # param = {"n_neighbors": [1, 3, 5]}
    # 使用网格搜索
    # gs = GridSearchCV(knn, param_grid=param, cv=2)
    # 输入数据
    # gs.fit(x_train, y_train)
    # 得出测试集的准确率
    # print("测试集的准确率：", gs.score(x_test, y_test))
    # print("在交叉验证当中的最好验证结果：", gs.best_score_)
    # print("选择了模型：", gs.best_estimator_)
    # print("每个超参数每一个交叉验证：", gs.cv_results_)
    # return None


def navie_bayes():
    """
    朴素贝叶斯对新闻分类
    :return: None
    """
    # 获取新闻的数据集
    news = fetch_20newsgroups(subset='all')

    # 进行数据集的分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 进行特征抽取
    tf = TfidfVectorizer()

    x_train = tf.fit_transform(x_train)

    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯分类
    mlb = MultinomialNB(alpha=1.0)

    mlb.fit(x_train, y_train)

    y_predict = mlb.predict(x_test)

    print("预测的文章类型结果：", y_predict)

    score = mlb.score(x_test, y_test)

    print("准确率：", score)

    print(classification_report(y_test, y_predict, target_names=news.target_names))

    return None


if __name__ == "__main__":
    knncls()
    # tts1()
    # tts1()
    # tts2()
# 1、数据集    训练集 和 测试集
# 转换器和估计器
# 估计器流程：1、fit    2、predict,  score

# 2、k-近邻算法
# 1、距离公式   欧氏距离

# 3、朴素贝叶斯算法：朴素
# （2）条件概率和联合概率
# （3）贝叶斯公式 P( 类别 | 文档)

# 4、模型评估方法
# 准确率，精确率和召回率 F1

# 5、模型的选择
# 交叉验证：训练集+ 验证集    K折交叉验证  让算法更充分的去训练数据，模型得出的结果更加可信
# 网格搜索：超参数     自动的选择比较好的参数

# 6、决策树、随机森林
# 理解信息熵还有信息增益
# ID3            C4.5          CART
# 信息增益        信息增益比       基尼系数
# 随机森林 ：多个决策树       ---> 集成方法:利用多个分类器共同决定
# 又放回的随机抽样
