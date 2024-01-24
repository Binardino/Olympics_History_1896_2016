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

def add_lat_lon(df):
    """add latitude & longitude to df based on city name
    """
    #fetching actual locator from str city name
    locator = Nominatim(user_agent="myGeocoder")
    df['loc'] = df['City'].apply(lambda x : locator.geocode(x))
    #fetching only lat & long attributes
    df['latitude'] = df['loc'].apply(lambda x : x.latitude)

    df['longitude'] = df['loc'].apply(lambda x : x.longitude)

    #df.drop('loc', axis=1, inplace=True)

    return df

def create_slider_numeric(label, column, step):
    slider_numeric = st.sidebar.slider(label,
                                int(column.min()),
                                int(column.max()),
                                step)
    return slider_numeric

def create_slider_multiselect(label, column):
    slider_multiselect = st.sidebar.multiselect(label,
                                                column,
                                                column
                                                )
    return slider_multiselect