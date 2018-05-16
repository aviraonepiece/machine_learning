# -*- coding:utf-8 -*-



from sklearn.metrics import classification_report
import sys
import numpy as np
import nltk

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")

#训练集的样本操作序列数为N(从前往后数)，包含前50个正常的；测试集的样本数为150-N（从后往前数），共150个操作序列
N=100



def load_user_cmd(filename): #加载操作序列文件函数
    cmd_list=[]
    dist_max=[]
    dist_min=[]
    dist=[]
    # 下面代表着一次读取操作命令 每一百个组成一个操作序列，放到cmd_list
    with open(filename) as f: #作为 try 打开文件读取  finnlly  f.close的简洁写法
        i=0
        x=[]
        for line in f:
            line=line.strip('\n') #去掉开头结尾的换行
            x.append(line)
            dist.append(line)
            i+=1
            if i == 100:
                cmd_list.append(x)
                #print(x) 循环内的每个x都是一个操作序列，包含100条命令
                x=[]
                i=0
                #print(cmd_list) 只有一个cmd_list，且里面有15000/100=150个x（操作序列）0-149



    fist=dist[0:5000]  #dist里面由上面得出，包含15000个命令
    fdist = nltk.FreqDist(fist)#由于后面的要做特征重合度对比，个人认为统计正常操作习惯的最频繁最不频繁比较妥
    ser = pd.Series(fdist)
    sersort = ser.sort_values()  # 按照升序排列
    dist_min = sersort.index[0:50].tolist()  # 取出频率最小的前50个操作命令
    dist_max = sersort.index[-50:].tolist()  # 取出频率最大的最后50个操作命令
    return cmd_list, dist_max, dist_min

def get_user_cmd_feature(user_cmd_list,dist_max,dist_min):#user_cmd_list是150个操作序列，0-149
    user_cmd_feature=[]
    for cmd_block in user_cmd_list:
        f1=len(set(cmd_block)) #抽取每一个序列，对序列里的命令做去重操作，去重后的命令个数作为特征一


        fdist = nltk.FreqDist(cmd_block)
        ser = pd.Series(fdist)
        sersort = ser.sort_values()  # 按照升序排列
        f3 = sersort.index[0:10].tolist()  # 取出最不频繁的10个操作命令作为特征三
        f2 = sersort.index[-10:].tolist()  # 取出最频繁的10个操作命令作为特征二

        f2 = len(set(f2) & set(dist_max))   #和最频繁的50条命令计算重合个数
        f3=len(set(f3)&set(dist_min))       #和最不频繁的50条命令计算重合个数
        x=[f1,f2,f3]
        user_cmd_feature.append(x)
    return user_cmd_feature

def get_label(filename,index=0):
    x=[]

    with open(filename) as f:

        for line in f:
            line=line.strip('\n')#去掉换行符,否则会打印多一个回车
            x.append( int(line.split()[index]))#index为标记，第3个用户的标记是line[用户异常行为检测]，从0开始

    return x  #x为竖列对应的标记

if __name__ == '__main__':

    for usernum in range(1,51):

            user_cmd_list,user_cmd_dist_max,user_cmd_dist_min=load_user_cmd("D:/ml/用户异常行为检测/MasqueradeDat/User%s" % (usernum))#"./MasqueradeDat/User9"
            #此时最频繁的命令已经被统计，放入特征提取（数据清洗）
            user_cmd_feature=get_user_cmd_feature(user_cmd_list,user_cmd_dist_max,user_cmd_dist_min)
            #此时得到了特征集,共三个数值（特征提取）
            labels=get_label("D:/ml/用户异常行为检测/MasqueradeDat/label.txt",usernum-1)
            #此时得到了样本标签，是一个竖列标记
            y=[0]*50+labels #在lables[100]这个list从前插入50个0，意味着前面的50个操作序列是正常的，凑成完整的操作序列


            x_train=user_cmd_feature[0:N]   #被划分的样本特征集，训练集（命令数，与统计重合的最频繁命令数，最不频繁的命令数）（150个取N个，含50个正常）
            y_train=y[0:N]                  #被划分的样本标签 ，训练集(150个取N个，含50个正常）

            x_test=user_cmd_feature[N:150]  #测试集，样本特征集取后50个
            y_test=y[N:150]                 #测试集，样本标签取后50个

            #KNN
            neigh = KNeighborsClassifier(n_neighbors=6,algorithm='auto') #k值经过调整，设为6，方法自动选择，原来有三个方法
            neigh.fit(x_train, y_train)
            y_predict=neigh.predict(x_test)  #根据模型对测试集进行一个预测
            score=np.mean(y_test==y_predict)*100  #将的预测标记和已有的特征标记做对比，取均值
           # print ('User%s实际的后50个操作序列特征标签是(0为正常):' % (usernum),y_test)
            #print ('   KNN的预测后50个操作序列特征标签是(0为正常):',y_predict.tolist())
            print ('User %s KNN异常操作的预测准确率是：'%(usernum),score)
            target_name = ['正常', '异常']
           # print (classification_report(y_test, y_predict,target_names=target_name))

           # print(model_selection.cross_val_score(neigh, user_cmd_feature, y, n_jobs=-1, cv=10))
            y_predict_knn10=model_selection.cross_val_predict(neigh, user_cmd_feature, y, n_jobs=-1, cv=10)
            score = np.mean(y_test == y_predict_knn10[-50:]) * 100
            # 将的预测标记和已有的特征标记做对比，取均值，这里取150个的后50个序列（测试集序列）
        #    print('User%s实际的后50个操作序列特征标签是(0为正常):' % (usernum), y_test)
         #   print('十折交叉验证后50个操作序列特征标签是(0为正常):', y_predict_knn10[-50:].tolist())#同样取后50个测试集
            print('User %s KNN的十折交叉异常操作的预测准确率是：' %(usernum), score,'\n')




            # #SVM
            # clfsvm = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
            # y_predict_svm = neigh.predict(x_test)  # 根据模型对测试集进行一个预测
            # score = np.mean(y_test == y_predict_svm) * 100  # 将的预测标记和已有的特征标记做对比，取均值
            # print('SVM实际的后50个特征标签是（0为正常）:', y_test)
            # print('SVM的预测后50个特征标签是（0为正常）:', y_predict_svm.tolist())
            # print('SVM异常操作的预测准确率是：', score)
            # target_name = ['正常', '异常']
            # print(classification_report(y_test, y_predict_svm, target_names=target_name))
            #
            #
            # #NB
            # clfnb = GaussianNB().fit(x_train, y_train)
            # y_predict_nb = clfnb.predict(x_test)
            # score = np.mean(y_test == y_predict_nb) * 100  # 将的预测标记和已有的特征标记做对比，取均值
            # print('NB实际的后50个特征标签是（0为正常）:', y_test)
            # print('NB的预测后50个特征标签是（0为正常）:', y_predict_nb.tolist())
            # print('NB异常操作的预测准确率是：', score)
            # target_name = ['正常', '异常']
            # print(classification_report(y_test, y_predict_nb, target_names=target_name))

