# !/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import matplotlib as plot

reload(sys)
sys.setdefaultencoding('utf-8')


decisionnode = dict(boxstyle='sswtooth', fc='0.8')     # 决策节点
leafnode = dict(boxstyle='round4', fc='0.8')    # 叶子结点
arrow_args = dict(arrowstyle='<-')  # 箭头参数


# 绘制带箭头的注解
def plotnode(nodetxt, centerpt, parentpt, nodetype):
    createplot.axl.annotate(nodetxt, xy=parentpt, xycoords='axes fraction', xytext=centerpt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodetype, arrowprops=arrow_args)


