import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

df=pd.read_csv('heart.csv')
st.dataframe(df)
st.markdown("On propose en premier lieu de regarder quelques graphiques descriptive de notre base de donnée")

age_target=pd.crosstab(df.age,df.target)
sex_target=pd.crosstab(df.sex,df.target)
trestbps_target=pd.crosstab(df.trestbps,df.target)
chol_target=pd.crosstab(df.trestbps,df.target)
dic=age_target.to_dict()
dfc=pd.concat({k: pd.Series(v) for k, v in dic.items()}).reset_index()
dfc.columns = ['0_1', 'level','number']


gp_chart = alt.Chart(dfc).mark_bar().encode(
  alt.Column('0_1'), alt.X('level'),
  alt.Y('number', axis=alt.Axis(grid=False)), 
  alt.Color('Player'))


st.altair_chart(gp_chart, use_container_width=False)

plot_tab=pd.crosstab(df.age,df.target)
