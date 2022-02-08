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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier 


df=pd.read_csv('heart.csv')
st.dataframe(df)
st.markdown("We begin by presenting the dataset,  ")

variables_continues=['age','chol','trestbps','thalach','oldpeak']

a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]
df_dum = pd.concat(frames, axis = 1)
df_dum = df_dum.drop(columns = ['cp', 'thal', 'slope'])

y = df_dum.target.values
x_data = df_dum.drop(['target'], axis = 1)
#normalisation 
# x =x_data.copy()# (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
# for n in variables_continues:
#   x[n]=(x[n]-np.mean(x[n]))/np.std(x[n])
x=(x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
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
      self.algo=KNeighborsClassifier(algorithm='brute')
      self.grid_param= {'n_neighbors':[1,3,4,5,6,7,8,9,10,11,12,13],
              'leaf_size':[1,2,3,4,5,6,7,9],
               'weights':['uniform','distance'],         
              'algorithm':['auto', 'kd_tree']}
      #{'knn__n_neighbors': [3, 5, 7, 9, 11,15,17]}
#         'n_neigbours':[3,5,10,15],
#                        'weights':['uniform','distance'],
#                        'metric':['euclidean','manhattan']}
                       
     
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
      
    if str == 'Decision Tree':
      self.algo = DecisionTreeClassifier()
      self.grid_param={'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],}
      
    if str == 'Random Forest':
      self.algo = RandomForestClassifier()
      self.grid_param={'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],}
      
    if str == 'Perceptron':
      self.algo = Perceptron(tol=1e-3, random_state=0)
      self.grid_param={'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],}     
    
    if str == 'XGBoost':
      self.algo = XGBClassifier()
      self.grid_param={'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],}
    
    if str == 'Adaboost':
      self.algo = AdaBoostClassifier(n_estimators=50, random_state=0)
      self.grid_param={'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],}
      
  def train_classifieur(self,X_train,Y_train):
    self.algo.fit(X_train,Y_train)
    
  def scor_classifieur(self,X_test,Y_test):
    return(self.algo.score(X_test,Y_test)*100)
  
#   def Grid_search_CrossV(self,X_train,Y_train,score=False,best=False):
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#     search = GridSearchCV(self.algo,self.grid_param, scoring='accuracy', n_jobs=-1, cv=cv)
#     result = search.fit(X_train, Y_train)
#     if score==True:
#       return(result.best_score_)
#     if best==True:
#       return(result.best_params_)
      
     

      
      
      
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
     ('KNeighbors','Logistic Regression','Support Vector Machine Algorithm','Naive Bayes Algorithm','Decision Tree', 'Random Forest', 'Perceptron', 'XGBoost','Adaboost'))


# for i in range(1,20):
#     knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
#     knn2.fit(x_train.T, y_train.T)
#     scoreList.append(knn2.score(x_test.T, y_test.T))

choix_classifieur=classifieur(Model)
choix_classifieur.train_classifieur(x_train,y_train)

st.write("The precision of the standard model is :", choix_classifieur.scor_classifieur(x_test,y_test))
st.write("Whould you like to tune your model using grid ")
if Model in ['KNeighbors','Logistic Regression','Support Vector Machine Algorithm','Naive Bayes Algorithm']:
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
  search = GridSearchCV(choix_classifieur.algo,choix_classifieur.grid_param, scoring='accuracy', n_jobs=-1, cv=3)
  result = search.fit(x_train, y_train)
  st.write("The precision of the tuned model using grid searsh is :",100*result.best_score_)

##st.write("la préc",choix_classifieur.Grid_search_CrossV(x_train,y_train,score=True,best=False)

# pipe = Pipeline([
#         ('sc', StandardScaler()),     
#         ('knn', KNeighborsClassifier(algorithm='brute')) 
#     ])
#     params = {
#         'knn__n_neighbors': [3, 5, 7, 9, 11]
#     }
#     clf = GridSearchCV(estimator=pipe,           
#                       param_grid=params, 
#                       cv=5,
#                       return_train_score=True) 
#     clf.fit(x_train, y_train)


# mds_mean = np.zeros(13)
# N_mean = 300

# for n in range(N_mean):
#     mds = []
#     for md in range(2,15):
#         tree3 = DecisionTreeClassifier(max_depth = md)  
#         tree3.fit(X_train, y_train)
#         #tree3.score(X_test, y_test)
#         mds.append(tree3.score(X_test, y_test)) 
#     mds_mean += np.array(mds)
#     print(n)
    
# plt.plot([k for k in range(2,15)], 1/N_mean*mds_mean)
#plt.show()
