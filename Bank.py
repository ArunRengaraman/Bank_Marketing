#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st 
import numpy as np 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier



from sklearn.metrics import accuracy_score
import pandas as pd


# In[9]:




st.title('Streamlit Example')

st.write("""
# Explore different classifier and datasets
Which one is the best?
""")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",("Bank","Dummy"))

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('GB', 'SVM', 'Random Forest')
)


def get_dataset(name):
    data = None
    name == 'Bank Marketing'
    data = pd.read_csv('bank/bank-full.csv')
    objList = data.select_dtypes(include = "object").columns
    le = LabelEncoder()
    for values in objList:
        data[values] = le.fit_transform(data[values].astype(str))
        data = data
    X = data.iloc[:, 1:-1]
    y = data.iloc[:,-1]         
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'GB':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'GB':
        clf = GradientBoostingClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'],random_state=1234)
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

st.write('Enter the values for prediction')
age = st.number_input("AGE")
balance = st.number_input("BALANCE")
day = st.number_input("DAY")
duration = st.number_input("DURATION")
campaign = st.number_input("CAMPAIGN")
pdays = st.number_input("PDAYS")
previous = st.number_input("PREVIOUS")

# Categorical columns
job_options = ['management', 'technician', 'entrepreneur', 'blue-collar',
               'unknown', 'retired', 'admin.', 'services', 'self-employed',
               'unemployed', 'housemaid', 'student']
job = st.selectbox("JOB", options=job_options)

marital_options = ['married', 'single', 'divorced']
marital = st.selectbox("MARITAL", options=marital_options)

education_options = ['tertiary', 'secondary', 'unknown', 'primary']
education = st.selectbox("EDUCATION", options=education_options)

housing_options = ['yes', 'no']
housing = st.selectbox("HOUSING", options=housing_options)

loan_options = ['no', 'yes']
loan = st.selectbox("LOAN", options=loan_options)

contact_options = ['unknown', 'cellular', 'telephone']
contact = st.selectbox("CONTACT", options=contact_options)

month_options = ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'jan', 'feb',
                 'mar', 'apr', 'sep']
month = st.selectbox("MONTH", options=month_options)

poutcome_options = ['unknown', 'failure', 'other', 'success']
poutcome = st.selectbox("POUTCOME", options=poutcome_options)

client_data = [age, job, marital, education, default, balance, housing,
       loan, contact, day, month, duration, campaign, pdays,
       previous, poutcome]
data= np.array(list(client_data)).reshape(1,-1)

clf.predict(data)
if clf.predict(data)[0] == 1:
    st.write("The client subscribed a term deposit")
else:
    st.write("No subscribed a term deposit")
