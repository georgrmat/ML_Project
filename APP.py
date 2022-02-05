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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


df=pd.read_csv('heart.csv')
st.dataframe(df)
st.markdown("On propose en premier lieu de regarder quelques graphiques descriptive de notre base de donnée")

a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]
df_dum = pd.concat(frames, axis = 1)
df_dum = df_dum.drop(columns = ['cp', 'thal', 'slope'])

y = df_dum.target.values
x_data = df_dum.drop(['target'], axis = 1)
#normalisation 
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)


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
      
     
    
class classifieur:#(str):#,par,X_trai,Y_train,X_test,Y_test):
  
  def __init__(self,str):#,par,X_trai,Y_train,X_test,Y_test):
    if str=='KNeighbors':
      self.algo=KNeighborsClassifier(n_neighbors = 3)
      self.grid_param={'n_neigbours':[3,5,10,15],
                       'weights':['uniform','distance'],
                       'metric':['euclidean','manhattan']}
                       
     
    if str=='Logistic Regression':
      self.algo= LogisticRegression()
      self.grid_param={'solver':['newton-cg', 'lbfgs', 'liblinear'],
                        'penalty':['none', 'l1', 'l2', 'elasticnet'],
                      'C':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
      
    if str=='Support Vector Machine Algorithm':
      self.algo=SVC(random_state = 1)
      self.grid_param={'C': [0.1, 1, 10, 100, 1000],
                       'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                       'kernel': ['rbf']}
    if str=='Naive Bayes Algorithm':
      self.algo=GaussianNB() 
      self.grid_param={'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],}
      
      
  def train_classifieur(self,X_train,Y_train):
    self.algo.fit(X_train,Y_train)
    
  def scor_classifieur(self,X_test,Y_test):
    return(self.algo.score(X_test,Y_test)*100)
  
  def Grid_search_CrossV(self,X_train,Y_train,X_test,Y_test,score=False,best=False):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    search = GridSearchCV(self.algo,self.grid_param, scoring='accuracy', n_jobs=-1, cv=cv)
    result = search.fit(X_train, Y_train)
    if score:
      return(result.best_score_)
    if best:
      return(result.best_params_)
      
     

      
      
      
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
      ('Age', 'Cholesterol','ST depression induced by exercise relative to rest','Resting blood pressure','Maximum heart rate achieved'))
#('Age', 'Cholesterol', 'Sex','Chest pain','Resting blood pressure','Fasting blood sugar','Resting electrocardiographic results','Maximum heart rate achieved',
     #'Exercise induced angina','ST depression induced by exercise relative to rest','The slope of the peak exercise ST segment'))

var2 = st.radio(
     "What is the second variable that you chose?  ",
     ('Cholesterol','Age','ST depression induced by exercise relative to rest','Resting blood pressure','Maximum heart rate achieved'))
#('Age', 'Cholesterol', 'Sex','Chest pain','Resting blood pressure','Fasting blood sugar','Resting electrocardiographic results','Maximum heart rate achieved',
 #    'Exercise induced angina','ST depression induced by exercise relative to rest','The slope of the peak exercise ST segment'))


scater_var1_var2=alt.Chart(df).mark_point().encode(
    x=variable(var1).name,
    y=variable(var2).name,
    color=alt.Color('target',  scale=alt.Scale(scheme='dark2'),legend=alt.Legend(title="age VS chol"))
)

st.altair_chart(scater_var1_var2, use_container_width=False)

scoreList = []

Model=st.radio(
     "What is the model you want to use for the classification? ",
     ('KNeighbors','Logistic Regression','Support Vector Machine Algorithm','Naive Bayes Algorithm'))


# for i in range(1,20):
#     knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
#     knn2.fit(x_train.T, y_train.T)
#     scoreList.append(knn2.score(x_test.T, y_test.T))

choix_classifieur=classifieur(Model)
choix_classifieur.train_classifieur(x_train,y_train)

st.write("la précision de votre modèle est", choix_classifieur.scor_classifieur(x_test,y_test))

st.write("la meilleur précision de votre modèle est",choix_classifieur.Grid_search_CrossV(x_train,y_train,x_test,y_test,score=True)

