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

#%%#reading dfs
df_athlete = get_data_csv('../data/athlete_events.csv')

df_city_olympics = get_data_csv('../data/olympic_city_country.csv')

df_city = get_data_csv('../data/df_city.csv')
#sub df city for mapping
#adding countries to df_city
df_city = df_city.merge(df_city_olympics, how='left')

df_city = df_city.drop_duplicates().sort_values(by='Year').reset_index(drop=True)

st.write(df_city)
#%%
#create sliders
sidebar_years = create_slider_numeric('Olympics years',df_city.Year,4)

sidebar_countries = create_slider_multiselect('Countries',df_athlete.Team.unique())

sidebar_sports = create_slider_multiselect('Sports',df_athlete.Sport.unique())
#%%
st.markdown(""" Welcome to the ultimate Olympics EDA & Data Visualisation presentation
All of Modern Olympics data from 1896 to 2016 - 2020 Tokyo data pending """)
#%%


st.markdown("""Worldmap of Cities having held the Olympics""")
st.write("""Colors depends on Season - Size on amount of Olympics""")

#data wrangling
# Group by city and count the number of occurrences
map_data = df_city.groupby(['City']).agg({'Year'      : lambda x : x.tolist(),
                                          'Season'    : lambda x : x.unique(),
                                          'latitude'  : 'first',
                                          'longitude' : 'first',
                                          'Country'   :'count'}
                                         ).reset_index().rename(columns={'Country':'count'})

map_data['Year'] = map_data['Year'].apply(lambda x : str(x))

# map with px scatter mapbox
fig_map = px.scatter_mapbox(map_data, lat='latitude', lon='longitude', 
                            hover_data=['City','Year'],
                            size='count', color='Season', 
                            zoom=1, height=500,
                            title='Cities Hosting the Olympics',
                            mapbox_style='carto-positron')

# Customize the marker size and color scale
fig_map.update_traces(marker=dict(sizemode='area', sizeref=0.05), selector=dict(mode='markers'))

# Show the map
st.plotly_chart(fig_map)
#%%
# #creating 2 dfs for Summer & Winter Olympics
df_summer = df_city.loc[(df_city['Season'] == 'Summer')]

df_winter = df_city.loc[(df_city['Season'] == 'Winter')]
#%%
# #split per country / per medals
st.markdown("""Analysis of country performance per year""")
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
#%% Evolution of countries performance over time
# Group by country and year, and calculate the medal count - All Medal types
medal_counts = df_athlete.groupby(['Team', 'Year'])['Medal'].count().reset_index()

# Select the top performing countries
top_countries = medal_counts.groupby('Team')['Medal'].sum().nlargest(30).index

# Filter the data for the top performing countries
top_countries_data = medal_counts[medal_counts['Team'].isin(top_countries)].sort_values(by='Year')

# Create the animated bar chart
fig_evol = px.bar(top_countries_data, 
                    x='Team', y='Medal', 
                    animation_frame='Year', 
                    color='Team',
                    labels={'Team': 'Country', 
                            'Medal': 'Medal Count'}, 
                    # range_x=[0, top_countries_data.Team.nunique()],
                    range_y=[0, top_countries_data['Medal'].max()],
                    title='Top Performing Countries\' Medal Counts Over Time')

# Customize the layout
fig_evol.update_layout(xaxis={'categoryorder': 'total descending'})
fig_evol.update_xaxes(tickfont=dict(size=20))
fig_evol.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000

st.plotly_chart(fig_evol)

# fig_evol.show()
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

st.markdown(""" Age & Weight Distribution""")
df_sport = df_athlete.groupby(['Year', 'Season', 'Sport']).agg({'Sport':'count'}) \
                .rename(columns={'Sport':'Count'}) \
                .reset_index()

st.write(df_sport)
fig_sport = px.bar(df_sport, x=df_sport['Year'], y=df_sport['Count'], 
                    color='Sport',
                    )

st.plotly_chart(fig_sport)

st.write("""Age & Gender distribution per sports""")
selection_sport_athlete = st.selectbox('select viz library', ['plotly', 'seaborn'])
if selection_sport_athlete == 'plotly':
    fig_sport_athlete = px.box(df_athlete.dropna(subset=['Age']), x='Sport', y='Age', color='Sex')
    st.plotly_chart(fig_sport_athlete)
elif selection_sport_athlete == 'seaborn':
    fig_sport_athlete = plt.figure(figsize=(12,6))
    sns.boxplot(data=df_athlete.dropna(subset=['Age']), x='Sport', y='Age', hue='Sex')
    plt.xticks(rotation=90)
    plt.title('Age and Gender Representation in Sports')
    plt.xlabel('Sport')
    plt.ylabel('Age')
    plt.legend(title='Gender')
    st.pyplot(fig_sport_athlete)

st.markdown("""Presence of Sports per year""")

st.caption("""Scatter plot per year to map when did each sport start being represented at the Olympics ?""")
fig_sportyear = px.scatter(df_sport, x=df_sport['Year'], y=df_sport['Sport'], color='Sport')

st.plotly_chart(fig_sportyear)
#%%
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
#correlation matrix
st.markdown("""Correlation matrix""")
df_corr = df_athlete[['Weight','Height','Age']].corr()

df_corr = df_athlete[['Medal','Weight','Height','Age']].copy()

# df_corr = df_corr.dropna(axis=0).copy()

df_corr = df_corr.fillna(np.nan).copy()

st.write(df_corr)
st.markdown("""Correlation matrix""")
df_corr2 = df_corr.groupby(['Medal']).agg({'Weight':'mean',
                                                            'Height':'mean',
                                                            'Age':'mean'}
                                                            )


# st.write(df_corr)

fig_corr = plt.figure(figsize=(15,7))

sns.heatmap(df_corr2)

st.pyplot(fig_corr)

# Filter out rows without medal achievements
df_corr = df_corr[df_corr['Medal'].notnull()]

# Compute correlation coefficients
correlations = df_corr.corr(method='pearson')

# Visualize the correlation matrix using a heatmap
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



# Show the plot
# 
