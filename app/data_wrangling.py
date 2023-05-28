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
from functions import *
#set path for dynamic function import
sys.path.append("..")
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

#reading dfs
df_athlete = pd.read_csv('../data/athlete_events.csv')

df_city_olympics = pd.read_csv('../data/olympic_city_country.csv')

df_city_raw = df_athlete[['City', 'Year', 'Season']].copy().drop_duplicates()

df_city = add_lat_lon(df_city_raw)

print(df_city.head())

