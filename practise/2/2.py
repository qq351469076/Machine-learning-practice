# !/usr/bin/python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split  # 训练集划分测试集
from sklearn.datasets import fetch_20newsgroups
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

reload(sys)
sys.setdefaultencoding('utf-8')


# K-近邻算法实现入住
def knncls():
    data = pd.read_csv("./train.csv")
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")  # 缩小数据集范围, 加快运算
    time_value = pd.to_datetime(data['time'], unit='s')  # time转为时间戳
    time_value = pd.DatetimeIndex(time_value)  # 把时间格式转换成字典格式，获取年，月，日
    data['weekday'] = time_value.weekday  # 构造新的特征，可用的方法: weekday, day ,hour
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data = data.drop(['time'], axis=1)  # 字典删除无用的列, 因为增加了新特征
    place_count = data.groupby('place_id').aggregate(
        np.count_nonzero)  # 删除一些签到位置少的签到点, 这句话聚合place_id, 并且统计了place_id出现了多少次
    tf = place_count[place_count.row_id > 3].reset_index()  # 进行筛选, 这里面都是出现了三次以上的place_id
    data = data[data['place_id'].isin(tf.place_id)]  # 保留出现三次以上的place_id, 重新进行赋值
    y = data['place_id']  # 取出特征值和目标值
    x = data.drop(['place_id'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    knn = KNeighborsClassifier()  # estimaotr估计器流程
    knn.fit(x_train, y_train)
    score = knn.score(x_test, y_test)
    print score
    param = {"n_neighbors": [1, 3, 5]}
    # 使用网格搜索
    gs = GridSearchCV(knn, param_grid=param, cv=2)
    # 输入数据
    # gs.fit(x_train, y_train)
    # 得出测试集的准确率
    # print("测试集的准确率：", gs.score(x_test, y_test))
    # print("在交叉验证当中的最好验证结果：", gs.best_score_)
    # print("选择了模型：", gs.best_estimator_)
    # print("每个超参数每一个交叉验证：", gs.cv_results_)
    # return None


# 朴素贝叶斯对新闻分类
def navie_bayes():
    news = fetch_20newsgroups(subset='all')  # 获取新闻的数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)  # 进行数据集的分割
    tf = TfidfVectorizer()  # 进行特征抽取
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)  # 进行朴素贝叶斯分类, 拉普拉斯平滑系数防止乘积等于0失效
    mlb = MultinomialNB(alpha=1.0)
    mlb.fit(x_train, y_train)
    y_predict = mlb.predict(x_test)
    print("预测的文章类型结果：", y_predict)
    score = mlb.score(x_test, y_test)
    print("准确率：", score)
    print(classification_report(y_test, y_predict, target_names=news.target_names))


if __name__ == "__main__":
    # knncls()
    navie_bayes()


