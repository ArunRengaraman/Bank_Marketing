import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def app():
    st.title('Data Visualization')

    # Load Bank Marketing dataset
    # Replace 'bank-full.csv' with the actual path to your dataset file
    data = pd.read_csv('bank-full.csv', delimiter=';')

    # Bar plot of job distribution
    st.header('Bar Plot: Job Distribution')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='job', data=data)
    plt.xticks(rotation=45)
    plt.xlabel('Job')
    plt.ylabel('Count')
    plt.title('Job Distribution')
    st.pyplot()

    # Line plot of balance vs. age
    st.header('Line Plot: Balance vs. Age')
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='age', y='balance', data=data)
    plt.xlabel('Age')
    plt.ylabel('Balance')
    plt.title('Balance vs. Age')
    st.pyplot()

    # Scatter plot of balance vs. duration
    st.header('Scatter Plot: Balance vs. Duration')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='duration', y='balance', data=data, hue='y')
    plt.xlabel('Duration')
    plt.ylabel('Balance')
    plt.title('Balance vs. Duration')
    st.pyplot()

    # Heatmap of correlation matrix
    st.header('Heatmap: Correlation Matrix')
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    st.pyplot()
