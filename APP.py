import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


df=pd.read_csv('heart.csv')
st.dataframe(df)
st.markdown("On propose en premier lieu de regarder quelques graphiques descriptive de notre base de donnée")

a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)
df.head()
df = df.drop(columns = ['cp', 'thal', 'slope'])
df.head()

class variable (str):
  
  def __init__(self,str):
    
    if str=='Age':
      self.name='age'
    if str=='Cholesterol':
      self.name='chol'
    if str=='Sex':
      self.name='sex'
    if str=='Chest pain':
      self.name='cp'
    if str=='Resting blood pressure':
      self.name='trestbps'
    if str=='Fasting blood sugar':
      self.name='fbs'
    if str=='Resting electrocardiographic results':
      self.name='restecg'
    if str=='Maximum heart rate achieved':
      self.name='thalach'
    if str=='Exercise induced angina':
      self.name='exang'
    if str=='ST depression induced by exercise relative to rest':
      self.name='oldpeak'
    if str=='The slope of the peak exercise ST segment':
      self.name='slope'
      
     
    
class classifieur(str,par,X_trai,Y_train,X_test,Y_test):
  
  def __init__(self,str):
    if str=='KNeighbors'
    self.algo=KNeighborsClassifier(n_neighbors = par)
    if str=='Logistic Regression':
      self.algo= LogisticRegression()
    if str=='Support Vector Machine Algorithm':
      self.algo=SVC(random_state = 1)
    if str=='Naive Bayes Algorithm':
      self.algo=GaussianNB() 
      
    def train_classifieur(self,X_train,Y_train):
      self.algo.fit(X_train,Y_train)
    
    def scor_classifieur(self):
      return(self.algo.score(X_test.T,y_test.T)*100)
      
     

      
      
      
var = st.radio(
     "What is the variable that you chose?  ",
     ('Age', 'Cholesterol','ST depression induced by exercise relative to rest','Resting blood pressure','Maximum heart rate achieved'))
#, 'Sex','Chest pain','Resting blood pressure','Fasting blood sugar','Resting electrocardiographic results','Maximum heart rate achieved',
 #    'Exercise induced angina','ST depression induced by exercise relative to rest','The slope of the peak exercise ST segment'))

var_target=pd.crosstab(df[variable(var).name],df.target)

dicvar=var_target.to_dict()
dfvar=pd.concat({k: pd.Series(v) for k, v in dicvar.items()}).reset_index()
dfvar.columns = ['0_1', 'level','variable']


gp_chartvar = alt.Chart(dfvar).mark_bar().encode(
  alt.Column('0_1'), alt.X('level'),
  alt.Y('variable', axis=alt.Axis(grid=False)), 
  alt.Color('0_1'))

 
st.altair_chart(gp_chartvar, use_container_width=False)
  
var1 = st.radio(
     "What is the first variable that you chose?  ",
     ('Age', 'Cholesterol', 'Sex','Chest pain','Resting blood pressure','Fasting blood sugar','Resting electrocardiographic results','Maximum heart rate achieved',
     'Exercise induced angina','ST depression induced by exercise relative to rest','The slope of the peak exercise ST segment'))

var2 = st.radio(
     "What is the second variable that you chose?  ",
     ('Age', 'Cholesterol', 'Sex','Chest pain','Resting blood pressure','Fasting blood sugar','Resting electrocardiographic results','Maximum heart rate achieved',
     'Exercise induced angina','ST depression induced by exercise relative to rest','The slope of the peak exercise ST segment'))


scater_var1_var2=alt.Chart(df).mark_point().encode(
    x=variable(var1).name,
    y=variable(var2).name,
    color=alt.Color('target',  scale=alt.Scale(scheme='dark2'),legend=alt.Legend(title="age VS chol"))
)

st.altair_chart(scater_var1_var2, use_container_width=False)

scoreList = []

Model=st.radio(
     "What is the model you want to use for the classification? ",
     ('KNeighbors','LDA'))


# for i in range(1,20):
#     knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
#     knn2.fit(x_train.T, y_train.T)
#     scoreList.append(knn2.score(x_test.T, y_test.T))



