import pandas as pd
import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

df1 = pd.read_csv('data/feature.csv')
df2 = pd.read_csv('data/class.csv')

x = df1.values
y = df2.values
x = x[:,1:]
y = y[:,1:]
x = x.astype('float32')
y = y.astype('float32')
y = np.argmax(y,axis=1)

print(x.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=0)
print('data load finish.....')
#print(df2.columns)
#print(np.sum(y_train,axis=0))
#print(np.sum(y_test,axis=0))

clf = GradientBoostingClassifier(n_estimators=50,learning_rate=0.1,max_depth=5,verbose=1)
clf2 = OneVsRestClassifier(clf)
clf.fit(X_train,y_train)
joblib.dump(clf,'model/mutul_gbdt_clf.m')

y_ = clf.predict(X_test)

score = recall_score(y_test,y_)
#召回率
print(score)

