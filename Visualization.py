import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def app():
    st.title('Data Visualization')

    # Load Bank Marketing dataset
    try:
        data = pd.read_csv('bank-full.csv', delimiter=';')
    except FileNotFoundError:
        st.error("Dataset file 'bank-full.csv' not found. Please check the path.")
        return

    objList = data.select_dtypes(include="object").columns

    # Bar plot of job distribution
    st.header('Bar Plot: Job Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='job', data=data, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel('Job')
    ax.set_ylabel('Count')
    ax.set_title('Job Distribution')
    st.pyplot(fig)

    # Line plot of balance vs. age
    st.header('Line Plot: Balance vs. Age')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='age', y='balance', data=data, ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Balance')
    ax.set_title('Balance vs. Age')
    st.pyplot(fig)

    # Histogram with slider
    st.header('Histogram with Slider: Distribution of Age')
    bins = st.slider('Number of bins', min_value=5, max_value=50, value=20)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['age'], bins=bins, kde=True, ax=ax)
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Age')
    st.pyplot(fig)

    # Scatter plot of balance vs. duration
    st.header('Scatter Plot: Balance vs. Duration')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='duration', y='balance', data=data, hue='y', ax=ax)
    ax.set_xlabel('Duration')
    ax.set_ylabel('Balance')
    ax.set_title('Balance vs. Duration')
    st.pyplot(fig)

    # Encode categorical variables for heatmap
    le = LabelEncoder()
    for values in objList:
        data[values] = le.fit_transform(data[values].astype(str))

    # Heatmap of correlation matrix
    st.header('Heatmap: Correlation Matrix')
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)
