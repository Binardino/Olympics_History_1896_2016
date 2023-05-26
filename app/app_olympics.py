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
#from pandas_profiling import ProfileReport
from geopy.geocoders import Nominatim
#set path for dynamic function import
sys.path.append("..")
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

st.set_page_config(page_title="Olympics Athletes EDA presentation")
#%%
#import data
# @st.cache_data
def get_data_csv(path):
    return pd.read_csv(path)

# @st.cache_data
def get_data_sql(query, engine):
    return pd.read_sql(query=query, con=engine)

#%%
#reading dfs
df_athlete = get_data_csv('../data/athlete_events.csv')

df_city_olympics = get_data_csv('../data/olympic_city_country.csv')

st.write(df_city_olympics)

st.write(df_athlete)

#%%
#fig = sns.pairplot(df_athlete, hue="Medal")

#st.pyplot(fig)
#%%
#sub df city for mapping

df_city = df_athlete[['City', 'Year', 'Season']].copy()

st.write(df_city)

#adding countries to df_city
df_city = df_city.merge(df_city_olympics, how='left')

df_city = df_city.drop_duplicates().sort_values(by='Year').reset_index(drop=True)

st.write(df_city)

#fetching actual locator from str city name
locator = Nominatim(user_agent="myGeocoder")
df_city['loc'] = df_city['City'].apply(lambda x : locator.geocode(x))
#fetching only lat & long attributes
df_city['latitude'] = df_city['loc'].apply(lambda x : x.latitude)

df_city['longitude'] = df_city['loc'].apply(lambda x : x.longitude)

df_city.drop('loc', axis=1, inplace=True)
st.write(df_city)

st.map(df_city)

fig_map = px.scatter_mapbox(df_city, lat="latitude", lon="longitude", zoom=3)

fig_map.update_layout(mapbox_style="open-street-map")
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map)
#creating 2 dfs for Summer & Winter Olympics
df_summer = df_city.loc[(df_city['Season'] == 'Summer')]

df_winter = df_city.loc[(df_city['Season'] == 'Winter')]
#df_city.drop_duplicates().sort_values(by='Year').reset_index(drop=True)