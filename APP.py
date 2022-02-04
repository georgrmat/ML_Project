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

dica=age_target.to_dict()
dfa=pd.concat({k: pd.Series(v) for k, v in dica.items()}).reset_index()
dfa.columns = ['0_1', 'level','variable']

dictres=trestbps_target.to_dict()
dftres=pd.concat({k: pd.Series(v) for k, v in dictres.items()}).reset_index()
dftres.columns = ['0_1', 'level','variable']

dicsex=sex_target.to_dict()
dfsex=pd.concat({k: pd.Series(v) for k, v in dicsex.items()}).reset_index()
dfsex.columns = ['0_1', 'level','variable']

diccol=trestbps_target.to_dict()
dfcol=pd.concat({k: pd.Series(v) for k, v in diccol.items()}).reset_index()
dfcol.columns = ['0_1', 'level','variable']



gp_charta = alt.Chart(dfa).mark_bar().encode(
  alt.Column('0_1'), alt.X('level'),
  alt.Y('variable', axis=alt.Axis(grid=False)), 
  alt.Color('level'))

gp_charttres = alt.Chart(dftres).mark_bar().encode(
  alt.Column('0_1'), alt.X('level'),
  alt.Y('variable', axis=alt.Axis(grid=False)), 
  alt.Color('level'))

gp_chartsex = alt.Chart(dfsex).mark_bar().encode(
  alt.Column('0_1'), alt.X('level'),
  alt.Y('variable', axis=alt.Axis(grid=False)), 
  alt.Color('level'))



st.altair_chart(gp_charta, use_container_width=False)
st.altair_chart(gp_chartsex, use_container_width=False)

plot_tab=pd.crosstab(df.age,df.target)
