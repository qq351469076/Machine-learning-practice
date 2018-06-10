# !/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import numpy as np

reload(sys)
sys.setdefaultencoding('utf-8')



# 中文分词方法
import jieba
def cutword():
    # jieba分词
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    content1 = []
    content2 = []
    content3 = []

    # 循环取出分词的结果，放入列表，返回空格隔开的字符串
    for word in con1:
        content1.append(word)
    for word in con2:
        content2.append(word)
    for word in con3:
        content3.append(word)

    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3


# 文本特征抽取
from sklearn.feature_extraction.text import CountVectorizer
def textcv():
	# 根据特征列表进行筛选， 不小于最小值， 不大于最大值都会展现出来
	vector = CountVectorizer(min_df=2, max_df=2)
	# 先把特征抽取出来，然后对每一句进行处理, [u'dislike', u'is', u'life', u'like', u'long', u'python', u'short', u'too']
	# 第一句看特征列表里有没有, 没有的置为0, 有置为1
	res = vector.fit_transform(["life is short,i like python","life is too long,i dislike python"])
	print(vector.get_feature_names())
	print(res.toarray())



# 字典特征抽取
from sklearn.feature_extraction import DictVectorizer
def dictvec():
	# 默认参数sparse是开启， 开启状态为sparse矩阵类型， 关闭为二维数组类型
    dict = DictVectorizer(sparse=False) # 也可以关闭sparse, 用toarray()方式转换成二维数组
    data = dict.fit_transform([{'city': '北京', 'temperature':100}, {'city': '深圳', 'temperature':30}])
    print(dict.get_feature_names())	# 获取参数特征列表


# 利用sklearn库进行中文分词
def countvec():
    c1, c2, c3 = cutword()
    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names())
    print(data.toarray())



# tfidf  考量的是每个分词在文章里面的重要程度
from sklearn.feature_extraction.text import TfidfVectorizer
def tfidfvec():
    c1, c2, c3 = cutword()
    # 过滤某些不需要的词
    tf = TfidfVectorizer(stop_words=["所以", "明天"])
    data = tf.fit_transform([c1, c2, c3])
    print(tf.get_feature_names())
    print(data.toarray())


# 归一化
from sklearn.preprocessing import MinMaxScaler
def minmax():
	# 可以指定归一化的区间
    mm = MinMaxScaler(feature_range=(2, 3))
    data = mm.fit_transform([[90,2,10,40],[60,4,15,45],[75,3,13,46]])
    print(data)


# 标准化
from sklearn.preprocessing import StandardScaler
def stand():
    std = StandardScaler()
    data = std.fit_transform([[ 1., -1., 3.],[ 2., 4., 2.],[ 4., 6., -1.]])
    print(data)


# 处理缺失值
from sklearn.preprocessing import Imputer
def im():
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    data = imputer.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
    print(data)


# 特征选择, 删除所有低方差特征
from sklearn.feature_selection import VarianceThreshold
def variance():
	# 可以设置方差范围
    vt = VarianceThreshold(threshold=0.0)
    data = vt.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)


# PCA数据集降维
from sklearn.decomposition import PCA
def pca():
    pc = PCA(n_components=3)
    data = pc.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])
    print(data)


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


# 总结

# 机器学习：数据， 分析获得的规律，对未知数据进行预测

# 数据来源和类型：
# 类型：离散型：区间内不可分
#        连续型：区间内可分

# scikit-learn,UCI ,kaggle

# 特征值+目标值
# 特征工程；
#1、特征抽取：字典抽取：one-hot编码
# 文本抽取：CountVectorizer：通过词频技术，出现的不重复的单词（省略了单个字母），
# 如果是中文，要进行分词（省略了单个字）
# tfidf：重要性程度
# tf
#idf:逆文档频率log(总次数/当前文档次数)

#2、特征预处理：某些场景需要
# 特征缩放：归一化，标准化
# 缺失值：填充，删除

# 3、特征选择：防止数据冗余，嘈杂数据
#过滤式：删除低方差特征

# 4、降维：维度，数据都改变（运算过程不需要了解）

# 机器学习算法分类：
# 监督学习：特征值+目标值
#分类：
#回归：
#标注：
# 非监督学习：特征值
# 聚类算法
























