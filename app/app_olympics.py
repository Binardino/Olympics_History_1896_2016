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
from functions import *

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

df_city = get_data_csv('../data/df_city.csv')

st.write(df_city_olympics)

st.write(df_athlete)

#%%
#fig = sns.pairplot(df_athlete, hue="Medal")

#st.pyplot(fig)
#%%
#sub df city for mapping

# df_city = df_athlete[['City', 'Year', 'Season']].copy()

st.write(df_city)

#adding countries to df_city
df_city = df_city.merge(df_city_olympics, how='left')

df_city = df_city.drop_duplicates().sort_values(by='Year').reset_index(drop=True)

st.write(df_city)

st.map(df_city)

fig_map = px.scatter_mapbox(df_city, lat="latitude", lon="longitude", zoom=3)

fig_map.update_layout(mapbox_style="open-street-map")
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map)
#%%
# #creating 2 dfs for Summer & Winter Olympics
df_summer = df_city.loc[(df_city['Season'] == 'Summer')]

df_winter = df_city.loc[(df_city['Season'] == 'Winter')]

#medals per country
df_medals = df_athlete.groupby(['Team', 'Medal']).agg({'Medal':'count'})

df_medals.columns=['medal_count']
df_medals.reset_index(inplace=True)

df_medals.sort_values(by=['medal_count'], ascending=False, inplace=True)

fig_medal = plt.figure(figsize=(15,7))


sns.barplot(data=df_medals.nlargest(25,'medal_count'),
            x='Team',
            y='medal_count',
           hue='Medal')

st.pyplot(fig_medal)
#%%
#EDA over athletes
st.markdown("""EDA over athletes""")

st.markdown("""Age distribution""")
df_age = df_athlete[['Age','Height','Weight']].fillna(0).copy()
df_age['Age'] = df_age['Age'].astype(int)
st.write(df_age)

fig = px.histogram(df_age, x='Age')

st.plotly_chart(fig)

#%%
#split per gender
st.markdown("""Split per gender""")
df_male   = df_athlete.loc[df_athlete['Sex'] == 'M']
df_female = df_athlete.loc[df_athlete['Sex'] == 'F']

fig_gender = px.pie(df_athlete, values=df_athlete.Sex.value_counts())

st.plotly_chart(fig_gender)
#%%
#split per sport
st.markdown("""Distribution of sports""")
df_sport = df_athlete.groupby(['Year', 'Season', 'Sport']).agg({'Sport':'count'}) \
                .rename(columns={'Sport':'Count'}) \
                .reset_index()

st.write(df_sport)
fig_sport = px.bar(df_sport, x=df_sport['Year'], y=df_sport['Count'], color='Sport')

st.plotly_chart(fig_sport)