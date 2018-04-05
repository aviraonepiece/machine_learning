import pandas as pd
import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib #模型保存模块dump方法
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score#评估方法包
import lightgbm as lgb

df1 = pd.read_csv('data/feature.csv')
df2 = pd.read_csv('data/class.csv')

x = df1.values #注意values和index的区别
y = df2.values
x = x[:,1:]     #除去第一列的所有数
#y = y[:,1:]
x = x.astype('float32')#被划分的样本特征集
y = y.astype('float32')#被划分的样本标签


back = y[:,0] #取所有行第一个数
land = y[:,1] #取所有行第二个数，这里对应特征
neptune = y[:,2]
normal = y[:,3]
pod = y[:,4]
smurf = y[:,5]
teardrop = y[:,6]

print(x.shape) #shap函数，几行几列np.shape
print(normal.shape)

y = teardrop#先只针对teardrop预测

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=0)#验证集占训练集40%，随机种子（random_state)每次不一样
print('data load finish.....')

print(np.sum(y_train))
print(np.sum(y_test))     #没有交叉验证，

clf = GradientBoostingClassifier(n_estimators=50,learning_rate=0.1,max_depth=5,verbose=1)
#gbdt初始化（迭代最大次数太大过拟合太小欠拟合，步长，决策树的最大深度，输出日志
clf.fit(X_train,y_train) #训练
y_ = clf.predict(X_test)#预测

score = f1_score(y_test,y_) #预测准确率

print(score)

joblib.dump(clf,'model/teardrop_clf.m')











