import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def app():
    st.title('Data Visualization')

    # Load Bank Marketing dataset
    # Replace 'bank-full.csv' with the actual path to your dataset file
    data = pd.read_csv('bank-full.csv', delimiter=';')
    objList = data.select_dtypes(include="object").columns
   
    # Bar plot of job distribution
    st.header('Bar Plot: Job Distribution')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='job', data=data)
    plt.xticks(rotation=45)
    plt.xlabel('Job')
    plt.ylabel('Count')
    plt.title('Job Distribution')
    st.pyplot()

    # Line plot of balance vs. age
    st.header('Line Plot: Balance vs. Age')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='age', y='balance', data=data)
    plt.xlabel('Age')
    plt.ylabel('Balance')
    plt.title('Balance vs. Age')
    st.pyplot()


    # Histogram with slider
    st.header('Histogram with Slider: Distribution of Age')
    bins = st.slider('Number of bins', min_value=5, max_value=50, value=20)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['age'], bins=bins, kde=True)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Distribution of Age')
    st.pyplot()

    # Scatter plot of balance vs. duration
    st.header('Scatter Plot: Balance vs. Duration')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='duration', y='balance', data=data, hue='y')
    plt.xlabel('Duration')
    plt.ylabel('Balance')
    plt.title('Balance vs. Duration')
    st.pyplot()
    
    le = LabelEncoder()
    for values in objList:
        data[values] = le.fit_transform(data[values].astype(str))
        sdata = data
    # Heatmap of correlation matrix
    st.header('Heatmap: Correlation Matrix')
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    st.pyplot()
