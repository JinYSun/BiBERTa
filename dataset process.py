import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
 
data = pd.read_csv(r"C:\Users\BM109X32G-10GPU-02\Desktop\crossdata\cross.csv")
 
data["PCE"] = np.floor(data["PCE"])
#data["PCE"].where(data["income_cat"]<5, 5.0, inplace=True)


X =data.iloc[:,:2]
y = data['PCE']

ss=StratifiedShuffleSplit(n_splits=2,test_size=0.1) 

for train_index, test_index in ss.split(X, y):
    print("TRAIN_INDEX:", train_index, "TEST_INDEX:", test_index) 
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y[train_index], y[test_index] 
    
    print("X_train:",X_train)
    print("y_train:",y_train)
ss=StratifiedShuffleSplit(n_splits=2,test_size=0.1) 
da = pd.read_csv(r"C:\Users\BM109X32G-10GPU-02\Desktop\crossdata\cross.csv")

test = da.iloc[test_index,:]
test=test.reset_index(drop=True)
test.to_csv('J:\ADB\Discussion/test0.csv')
#test.to_csv('H:/qdf\qdf\dataset/transf/test0.txt',index=False)
train1 = da.iloc[train_index,:]
da = da.iloc[train_index,:]
da=da.reset_index(drop=True)
# da=da.reset_index(drop=True)

train1=train1.reset_index(drop=True)
train1["PCE"] = np.floor(train1["PCE"])

X =train1.iloc[:,:2]
y = train1['PCE']


for train_index, val_index in ss.split(X, y):
    print("TRAIN_INDEX:", train_index, "TEST_INDEX:", val_index) 
    X_train, X_val =  X.iloc[train_index], X.iloc[val_index] 
    y_train, y_val = y[train_index], y[val_index] 
    
    print("X_train:",X_train)
    print("y_train:",y_train)
    
train=da.iloc[train_index,:]
train=train.reset_index(drop=True)
train.to_csv('J:\ADB\Discussion//train0.csv')
#train.to_csv('H:/qdf\qdf\dataset/transf//train0.txt',index=False)
val=da.iloc[val_index,:]
val=val.reset_index(drop=True)
val.to_csv('J:\ADB\Discussion//val0.csv')
#val.to_csv('H:/qdf\qdf\dataset/transf/val0.txt')