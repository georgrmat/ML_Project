import numpy as np
import pandas as pd
import streamlit as st

df=pd.read_csv('heart.csv')
st.dataframe(df)
st.markdown("On propose en premier lieu de regarder quelques graphiques descriptive de notre base de donnée")

J'ai modifié ça 

plot_tab=pd.crosstab(df.age,df.target)
