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

st.markdown("""Presence of Sports per year""")

st.caption("""Scatter plot per year to map when did each sport start being represented at the Olympics ?""")
fig_sportyear = px.scatter(df_sport, x=df_sport['Year'], y=df_sport['Sport'], color='Sport')

st.plotly_chart(fig_sportyear)
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

#%% Evolution of countries performance over time
# Group by country and year, and calculate the medal count
medal_counts = df_athlete.groupby(['NOC', 'Year'])['Medal'].count().reset_index()

# Select the top performing countries
top_countries = medal_counts.groupby('NOC')['Medal'].sum().nlargest(10).index

# Filter the data for the top performing countries
top_countries_data = medal_counts[medal_counts['NOC'].isin(top_countries)]

# Create the animated bar chart
fig_evol = px.bar(top_countries_data, x='NOC', y='Medal', animation_frame='Year', color='NOC',
             labels={'NOC': 'Country', 'Medal': 'Medal Count'}, 
             title='Top Performing Countries\' Medal Counts Over Time')

# Customize the layout
fig_evol.update_layout(xaxis={'categoryorder': 'total descending'})
fig_evol.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000

# Show the plot
st.plotly_chart(fig_evol)
