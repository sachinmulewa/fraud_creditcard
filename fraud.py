import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv (r"C:\Users\Asus\OneDrive - IIT Kanpur\Desktop\fraudTrain.csv")
print(df.shape)
print(df.info())
print(df["job"].nunique())
df.drop(["job","dob","trans_num","street","Unnamed: 0","state","city","last","first","trans_date_trans_time","category","merchant"],inplace= True,axis=1)
print(df.shape)
print(df.info())
print((df["gender"].unique()))
df["gender"]=df["gender"].map({"F":0,"M":1})
df["gender"]=df["gender"].astype(int)
print(df.info())
print(df.head().to_string() )
X=df.iloc[:,:9]
y=df.iloc[:,-1]
print(X.shape)
print(y.shape)
print(X.info())
print(y.info())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size= 0.8)
log=LogisticRegression ()
log.fit(X_train ,y_train )
y_pred=log.predict(X_test)
print(y_pred)
#y_pred.columns=["predicted"]
#print(y_pred)
y_pred=pd.DataFrame (y_pred)
y_pred.columns=["predicted"]
tree= DecisionTreeClassifier ()
tree.fit(X_train ,y_train )
y_pred_tree=tree.predict(X_test)
print(y_pred_tree)
#y_pred.columns=["predicted"]
#print(y_pred)
y_pred_tree=pd.DataFrame (y_pred_tree)
y_pred_tree.columns=["predicted"]
rftree= RandomForestClassifier  ()
rftree.fit(X_train ,y_train )
y_pred_rftree=rftree.predict(X_test)
print(y_pred_rftree)
#y_pred.columns=["predicted"]
#print(y_pred)
y_pred_rftree=pd.DataFrame (y_pred_rftree)
y_pred_rftree.columns=["predicted"]
from sklearn .metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred )
print("accuracy=",accuracy )
accuracy_tree=accuracy_score(y_test,y_pred_tree )
print("accuracy=",accuracy_tree)
accuracy_rftree=accuracy_score(y_test,y_pred_rftree )
print("accuracy=",accuracy_rftree)





#print(df["dob"].unique())
