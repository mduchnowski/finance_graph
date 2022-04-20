# M.Duchnowski
# Exploring Network Graphs
# April, 2022 


# --- import libraries
import numpy as np
import pandas as pd
import datetime
import json
import re

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

import streamlit as st

st.set_page_config(layout="wide")

#from sklearn import datasets, linear_model
#from sklearn.metrics import mean_squared_error, r2_score

pio.templates.default="ggplot2"

# --- define constants
today = datetime.date.today()

# Function for loading data into memory 
def load_data():
    
    df = pd.read_csv("data_capstone_dsa2021_2022.csv")
    
    # Calculate Item Stats - This includes only PPLUSES
    dfStats = df.filter(regex='^gs_(\d*)',axis=1).mean(axis=0).to_frame().T.add_suffix('_PPlus')
    
    #Every record will contain PPluses for easy access later
    df = df.merge(dfStats, how="cross")
    
    return df, dfStats

# Custom function for interpretting response to state 
@st.cache

#########################################################
# Exploration of Feature: Zip Score
#########################################################
st.header("Exploration of Netowk Grpahs")
st.markdown("Please send questions and  comments to: [mduchnowski@ets.org](mduchnowski@ets.org)")



colC1, colC2 = st.columns((5,5))


# Raw Score Distribution
figCohort1 = px.histogram(df, 
                          x="sum_score", nbins=20,  barmode='overlay', marginal="box", color="sum_cohort", 
                          category_orders={"sum_cohort": ["01-Low", "02-Mid", "03-High"]}, 
                          labels={"sum_cohort": "Raw Ability Cohort"}, 
                          title="<b>Raw Score Distribution</b>")