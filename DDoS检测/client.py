import pandas as pd
from sklearn.externals import joblib
import time

class DDos_Dection:
    def __init__(self):
        self.normal_clf = joblib.load('model/normal_clf.m')
        self.back_clf = joblib.load('model/back_clf.m')
        self.land_clf = joblib.load('model/land_clf.m')
        self.neptune_clf = joblib.load('model/neptune_clf.m')
        self.pod_clf = joblib.load('model/pod_clf.m')
        self.sumrf_clf = joblib.load('model/sumrf_clf.m')
        self.teardrop_clf = joblib.load('model/teardrop_clf.m')

        #加载clf
        self.clf = []
        self.ddos_type = ['back', 'land', 'neptune', 'pod', 'sumrf', 'teardrop']
        self.clf.append(self.back_clf)
        self.clf.append(self.land_clf)
        self.clf.append(self.neptune_clf)
        self.clf.append(self.pod_clf)
        self.clf.append(self.sumrf_clf)
        self.clf.append(self.teardrop_clf)

    def dection(self,data):
        type1 = []
        normal_flag = self.normal_clf.predict(data) #先看是不是正常
        if normal_flag==1:
            type1.append('normal')
        elif normal_flag==0:                        #再看是哪种ddos
            ddos_type = []
            for i in range(6):
                temp = self.clf[i].predict(data)
                if temp==1:
                    ddos_type.append(self.ddos_type[i])
            type1 = ddos_type
        return type1



def main():
    data = pd.read_csv('data/feature.csv')
    data = data.values
    data = data[:,1:] #取出所有数据的第二列到最后一列的数据
    clf = DDos_Dection()  # 加载DDos检测类
    for i in range(500):   #先测试个500条
        feature = data[i]
        #print('load data finish!')
        #print('load dection finish!')
        print('feature:')
        print(feature)
        result = clf.dection(feature)
        print('dect result :',result)
        time.sleep(5)



if __name__ == '__main__':
    main()