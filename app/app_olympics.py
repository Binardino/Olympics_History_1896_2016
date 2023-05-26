# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import sys
#set path for dynamic function import
sys.path.append("..")
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

st.set_page_config(page_title="Olympics Atheltes EDA presentation")
#%%
#import data
# @st.cache_data
def get_data_csv(path):
    return pd.read_csv(path)

# @st.cache_data
def get_data_sql(query, engine):
    return pd.read_sql(query=query, con=engine)

#%%
#read df
df_athlete = get_data_csv('../data/athlete_events.csv')

df_city_olympics = get_data_csv('../data/olympic_city_country.csv')

st.write(df_city_olympics)


st.write(df_athelete)

#%%
fig = sns.pairplot(df_athelete, hue="Medal")

st.pyplot(fig)
#%%
#sub df city for mapping

df_city = df_athlete[['City', 'Year', 'Season']].copy()

df_city = df_city.drop_duplicates().sort_values(by='Year').reset_index(drop=True)

st.write(df_city)