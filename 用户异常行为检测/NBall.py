# -*- coding:utf-8 -*-

import sys

import re

import numpy as np
from sklearn.metrics import classification_report
import sys
import nltk
import csv
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection

##训练集的样本操作序列数为N(从前往后数)，包含前50个正常的；测试集的样本数为150-N（从后往前数），共150个操作序列
N=100

def load_user_cmd_all(filename):#加载操作序列文件函数
    cmd_list=[]
    dist=[]
    with open(filename) as f:# 下面代表着一次读取操作命令 每一百个组成一个操作序列，放到cmd_list
        i=0
        x=[]
        for line in f:
            line=line.strip('\n')
            x.append(line)
            dist.append(line)
            i+=1
            if i == 100:
                cmd_list.append(x)
                #print(x) 每个x都是一个操作序列，包含100条命令
                x=[]
                i=0
                # print(cmd_list) 只有一个cmd_list，且里面有15000/100=150个x（操作序列）0-149

    fdist = list(FreqDist(dist).keys()) #dist由上面得出，包含15000个命令，这一步是去重
    return cmd_list,fdist        #cmd_list有150个操作序列，fdist是去重后的命令集合

def get_user_cmd_feature_all(user_cmd_list, dist):#传入上面函数返回的两个值，user_cmd_list是150个操作序列，dist上同fdist
    user_cmd_feature=[]

    for cmd_list in user_cmd_list:      #，150个序列，迭代150次 cmd_list,一个cmd_list代表一个序列,含100条命令
        v=[0]*len(dist)                 #dist是fdist，值依照不同用户而不同，初始化向量为全零，User3 的v为107个

        for i in range(0,len(dist)): #对于107个向量的分量进行迭代
            if dist[i] in cmd_list:  #如果向量的第i个分量（命令）在本次的cmd_list中被找到
                v[i]+=1                #对应的分量+1
        user_cmd_feature.append(v)   #user_cmd_feature收纳对应本次cmd_list序列的向量

    return user_cmd_feature     #对应有150个向量，每个向量有len(fdist)（去重后命令个数）个分量，user3为107个

def get_label(filename,index=0):
    x=[]
    with open(filename) as f:  #作为 try 打开文件读取  finnlly  f.close的简洁写法
        for line in f:
            line=line.strip('\n')
            x.append( int(line.split()[index]))# x对应lable.txt的一个竖列，代表一个用户的所有操作序列标签
    return x

if __name__ == '__main__':
    arg = sys.argv
    try:
        if len(arg) >= 2 and 0 < int(arg[1]) < 51:  # 51代表是50个用户

            usernum = int(arg[1])
            user_cmd_list,dist=load_user_cmd_all("D:/ml/用户异常行为检测/MasqueradeDat/User%s" % (usernum))#dist为去重后的序列
            print  ("该用户的去重向量表Dist:(%s)" % dist)
            user_cmd_feature=get_user_cmd_feature_all(user_cmd_list, dist)#150个向量，每个向量有len（dist）个分量，1或0表示

            labels=get_label("D:/ml/用户异常行为检测/MasqueradeDat/label.txt",usernum-1)
            y=[0]*50+labels  #加上前50个正常的序列标签

            x_train=user_cmd_feature[0:N]#取前N（100）个训练集（序列向量，样本特征集）
            y_train=y[0:N]                 #取前N个对应的样本特征标签

            x_test=user_cmd_feature[N:150]#测试集特征集
            y_test=y[N:150]                 #测试集特征标签



            clf=GaussianNB().fit(x_train,y_train)
            y_predict=clf.predict(x_test)
            score=np.mean(y_test==y_predict)*100


            print('User%s实际的后50个操作序列特征标签是(0为正常):' % (usernum), y_test)
            print('   NB预测的后50个操作序列特征标签是(0为正常):', y_predict.tolist())
            print('NB异常操作预测的准确率是：', score)
            target_name = ['正常', '异常']
            print(classification_report(y_test, y_predict, target_names=target_name))
            print( model_selection.cross_val_score(clf, user_cmd_feature, y, n_jobs=-1,cv=10))


            y_predict_nb10 = model_selection.cross_val_predict(clf, user_cmd_feature, y, n_jobs=-1, cv=10)
            score = np.mean(y_test == y_predict_nb10[-50:]) * 100
            # 将预测的标记和已有的特征标记做对比，取均值，这里取150个的后50个序列（测试集序列）
            print('User%s实际的后50个操作序列特征标签是(0为正常):' % (usernum), y_test)
            print('十折交叉验证后50个操作序列特征标签是(0为正常):', y_predict_nb10[-50:].tolist())  # 同样取后50个测试集
            print('KNN的十折交叉异常操作预测的准确率是：', score)


            try:
                if len(arg)==3 and 0<int(arg[2])<51:#51代表测试集有50个序列
                    print('\n')
                    bashlistnum = int(arg[2])-1
                    print('用户User%s测试集的第%d个操作序列是:' %(usernum,bashlistnum+1),user_cmd_list[100+bashlistnum])
                    print('该操作序列被NB预测的结果是:','0（正常，是该用户的操作）' if y_predict.tolist()[bashlistnum]==0 else '1（异常，不是该用户的操作）')
                    print('该操作序列被NB十折交叉预测的结果是:', '0（正常，是该用户的操作）' if y_predict_nb10[-50:].tolist()[bashlistnum]==0 else '1（异常，不是该用户的操作）')
                    print("该操作序列实际的结果是", '0（正常，是该用户的操作）' if y_test[bashlistnum]==0 else '1（异常，不是该用户的操作）')
            except:
                print('Usage:python %s Usernumber（0-50） [BashListNumber(0-50)]' % (arg[0]))

    except:
        print('Usage:python %s Usernumber（0-50） [BashListNumber(0-50)]' % (arg[0]))