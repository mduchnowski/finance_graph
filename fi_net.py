# M.Duchnowski
# Exploring Network Graphs
# April, 2022 


# --- import libraries
import numpy as np
import pandas as pd
import datetime
import json
import re

import pyvis
#from pyvis.network import Network
import networkx as nx

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

ContractColors = {'DAR': "#e60049", 'Item Development' : "#0bb4ff", 'NPD' : "#50e991", 'SDC' : "#e6d800", 'MDPS' : "#9b19f5", 'WTDOM' : "#ffa300", 'NSSC' : "#dc0ab4", 'PC' : "#b3d4ff", 'PSM' : "#00bfa0"}


# Function for loading data into memory 
def load_data():

    #Import Matrix
    x = pd.read_excel('Alliance list of high-level tasks_rev4-12.xlsx', skiprows=1, sheet_name='matrix concept', index_col=None)  
    x['Contract'].fillna(method='ffill', inplace=True)
    x.fillna('', inplace=True)

    #Derive numeric Group varaible
    x['Group'] = x['Contract'].rank(method='dense').astype(int)-1

    #Import Matrix
    y=pd.read_excel('Alliance list of high-level tasks_rev4-12.xlsx', skiprows=1, sheet_name='List of activities', index_col=None, header=None)  
    y[0].fillna(method='ffill', inplace=True)
    y = y[:61] #Clip the end of the data
    y.columns =['Contract', 'Activity', 'Dollars', 'PCT', 'PCT_TOT']

    df = pd.merge(x, y,  how='left', left_on=['Contract','Activity'], right_on = ['Contract','Activity'])
    
    return df

df = load_data()

#########################################################
# Data Illustrations for Allianc
#########################################################
st.header("Data Illustrations for Alliance")
st.markdown("Please send questions and  comments to: [mduchnowski@ets.org](mduchnowski@ets.org)")


#########################################################
# Financial Treemap
#########################################################
st.header("Financial Treemap")
st.markdown("Treemap Illustrating Contracts and Activities")

# Treemap
fig = px.treemap(df, path=['Contract','Activity'], values='PCT_TOT', color='Contract', width = 900, height = 900, hover_data=['Activity', 'PCT_TOT'], 
                 color_discrete_map=ContractColors)
fig.update_layout(uniformtext=dict(minsize=20))
fig.update_traces(legendgrouptitle_font_size=25, selector=dict(type='treemap'))



st.plotly_chart(fig)


#########################################################
# Network Graph
#########################################################
st.header("Network Graphs")
st.markdown("Netork illustrations based on incidence matrix")

# Initialize Graph
nx_graph =nx.Graph()

# Add nodes 
for index, row in df.iterrows():
    nx_graph.add_node(index, size= row['Dollars'], title=row['Contract'], label=row['Activity'], group=row['Group'])

# Add Edges 
for i in range(1,60):
    for j in range(i+1,60):
        value = df.iloc[i,j] 
        if value != '':
            nx_graph.add_edge(i-1, j-1)  

            
#Position for layout
pos = nx.circular_layout(nx_graph)
#pos = nx.random_layout(nx_graph)

#https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data
color_pallete = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]

color_map = []
legendtexts = []
for group in nx_graph.nodes(data="group"):
    color_map.append(color_pallete[group[1]])

for title in nx_graph.nodes(data="title"): 
    if title[1] not in legendtexts:
        legendtexts.append(title[1])
        
        

nx.draw(nx_graph, pos, node_color=color_map, node_size=200, with_labels=False, font_size=8)

fig = plt.gcf()
fig.set_size_inches(8, 8)

#Generate a legend
patches = [plt.plot([],[], marker="o", ms=13, ls="", mec=None, color=color_pallete[i], 
            label="{:s}".format(legendtexts[i]) )[0]  for i in range(len(legendtexts)) 
          ]

plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(1.2, 0.7), ncol=1, facecolor="white", numpoints=1, fontsize=15)

colC1, colC2 = st.columns((5,5))
colC1.pyplot(fig)
