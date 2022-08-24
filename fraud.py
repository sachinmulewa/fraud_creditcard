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
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size= 0.8)
log=LogisticRegression ()
log.fit(X_train ,y_train )
y_pred=log.predict(X_test)
print(y_pred)
#y_pred.columns=["predicted"]
#print(y_pred)
y_pred=pd.DataFrame (y_pred)
y_pred.columns=["predicted"]
from sklearn .metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred )
print("accuracy=",accuracy )




#print(df["dob"].unique())
