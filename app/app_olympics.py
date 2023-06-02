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

#%%
# #split per medals

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


fig_px_medal = px.bar(df_medals.nlargest(50,'medal_count'), 
                    x='Team', y='medal_count', 
                    color='Medal', color_discrete_sequence=("#FFD700", #gold
                                                            "#C0C0C0", #silver
                                                            "#CD7F32" #bronze
                                                            )
                    )

st.plotly_chart(fig_px_medal)

#split per sport
df_medalsport = df_athlete.groupby(['Sport', 'Medal']).agg({'Medal':'count'}).rename(columns={'Medal':'medal_count'}).reset_index()

fig_px_medalsport = px.bar(df_medalsport.nlargest(50,'medal_count'), 
                    x='Sport', y='medal_count', 
                    color='Medal', color_discrete_sequence=("#FFD700", #gold
                                                            "#C0C0C0", #silver
                                                            "#CD7F32" #bronze
                                                            )
                    )
st.plotly_chart(fig_px_medalsport)

#%%
#EDA over athletes
st.markdown("""EDA over athletes""")

st.markdown("""Age distribution""")
df_age = df_athlete[['Age','Height','Weight', 'Sex', 'Year']].dropna().copy()
df_age[['Age','Height','Weight']] = df_age[['Age','Height','Weight']].astype(int)
st.write(df_age)

fig_age = px.histogram(df_age, x='Age', color='Sex',barmode='group')

st.plotly_chart(fig_age)

st.markdown("""Weight distribution""")
st.write(df_age)

fig_weight = px.histogram(df_age, x='Weight', color='Sex',barmode='overlay')

st.plotly_chart(fig_weight)

fig_sex = px.histogram( df_age, x='Year', color='Sex',barmode='stack')

st.plotly_chart(fig_sex)

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

st.markdown("""Presence of Sports per year

When did each sport start being represented at the Olympics ?""")
fig_sportyear = px.scatter(df_sport, x=df_sport['Year'], y=df_sport['Sport'], color='Sport')

st.plotly_chart(fig_sportyear)
#%%
