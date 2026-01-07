import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df=pd.read_csv("/Users/samarthgarg/Downloads/ML ETP/DataSets/Social_Network_Ads.csv")

models={
    'Logistic': LogisticRegression(),
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(),
    'SVM': SVC(kernel='rbf')
}

X=df.drop('Purchased',axis=1)
y=df['Purchased']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

results=[]

for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    results.append({
        'Model':name,
        'Accuracy': round(acc,4)
    })

print(results)