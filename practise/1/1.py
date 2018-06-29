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
from sklearn.model_selection import GridSearchCV  # 网格搜索, 交叉验证

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
    print(vector.get_feature_names())  # 去重特征值
    print(res)  # 文本出现次数的向量


# 字典特征抽取
def dictvec():
    # 默认参数sparse是开启， 开启状态为sparse矩阵类型， 关闭为二维数组类型
    dict = DictVectorizer(sparse=False)  # 也可以关闭sparse, 用toarray()方式转换成二维数组
    res = dict.fit_transform([{'city': '北京', 'temperature': 100}, {'city': '深圳', 'temperature': 30},
                              {'city': '大庆', 'temperature': 50}])
    print(' '.join(dict.get_feature_names()))  # 去重文本特征值
    print(res)  # 字典文本出现次数的向量


# tfidf  考量的是每个分词在文章里面的重要程度
def tfidfvec():
    c1 = cutword()
    tf = TfidfVectorizer(stop_words=["所以", "明天"])  # 过滤某些不需要的词
    data = tf.fit_transform([c1])
    print(' '.join(tf.get_feature_names()))
    print(data.toarray())


# 归一化
def minmax():
    mm = MinMaxScaler(feature_range=(0, 1))  # 可以指定归一化的区间, 默认在[0, 1]之间
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
    print(li.target_names)  # 获取目标值标签
    """训练集的特征值       测试集的特征值     训练集类别     测试集类别,      规模为0.25, 随机种子自定"""
    x_train, y_test, x_target, y_target = train_test_split(li.data, li.target, test_size=0.25, random_state=24)
    print(x_train, y_test)


# 网格搜索, 交叉验证 + kNN算法
def gridsearchcv():
    x = '假装是训练集'
    y = '假装是训练集目标值'
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)  # 划分测试集
    std = StandardScaler()  # 标准化
    x_train = std.fit_transform(x_train)  # 标准化转化规则
    x_test = std.transform(x_test)  # 测试集采用转化规则
    knn = KNeighborsClassifier()  # 初始化kNN估计器
    knn.fit(x_train, y_train)  # 拟合数据, 转化规则
    score = knn.score(x_test, y_test)
    print score
    param = {"n_neighbors": [1, 3, 5]}  # 网格搜索参数
    gs = GridSearchCV(knn, param_grid=param, cv=2)  # cv参数是交叉验证的次数, 取多少次
    gs.fit(x_train, y_train)
    print("测试集的准确率：", gs.score(x_test, y_test))
    print("选择了模型：", gs.best_estimator_)
    print("每个超参数每一个交叉验证：", gs.cv_results_)
    print("在交叉验证当中的最好验证结果：", gs.best_score_)  # 返回最佳的准确率 0.8211334476626017
    print(gs.best_params_)  # 返回最佳的参数  {'n_estimators': 60}


# 朴素贝叶斯
def navie_bayes():
    news = fetch_20newsgroups(subset='all')  # 获取新闻的数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)  # 进行数据集的分割
    tf = TfidfVectorizer()  # 进行特征抽取
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    mlb = MultinomialNB(alpha=1.0)  # 进行朴素贝叶斯分类, 拉普拉斯平滑系数防止乘积等于0失效
    mlb.fit(x_train, y_train)
    y_predict = mlb.predict(x_test)
    print("预测的文章类型结果：", y_predict)
    score = mlb.score(x_test, y_test)  # 朴素贝叶斯估计器
    print("准确率：", score)
    print(classification_report(y_test, y_predict, target_names=news.target_names))  # 精确度和召回率和fi


if __name__ == "__main__":
# dictvec()
# textcv()
# countvec()
# tfidfvec()
# minmax()
# stand()
# im()
# variance()
# pca()
# cutword()
# tts()

# 总结
# -------------------------------------------------第一天--------------------------------------------------------------
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

# -----------------------------------------------第二天----------------------------------------------------------------
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