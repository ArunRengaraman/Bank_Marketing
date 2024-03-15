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
    "Select Dataset",("Bank Marketing","Dummy"))

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('GB', 'SVM', 'Random Forest')
)


def get_dataset(name):
    data = None
    name == 'Bank Marketing'
    data = pd.read_csv('bank-full.csv', delimiter=";")
    objList = data.select_dtypes(include = "object").columns
    le = LabelEncoder()
    for values in objList:
        data[values] = le.fit_transform(data[values].astype(str))
        data = data
    X = data.loc[:, data.columns != 'y']
    y = data.loc[:, data.columns == 'y']
    return X, y

X, y = get_dataset(dataset_name)

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
# Define mapping of numeric codes to categories for each categorical variable
job_options = {0: 'admin.', 1: 'blue-collar', 2: 'entrepreneur', 3: 'housemaid',
               4: 'management', 5: 'retired', 6: 'self-employed', 7: 'services',
               8: 'student', 9: 'technician', 10: 'unemployed', 11: 'unknown'}

marital_options = {0: 'divorced', 1: 'married', 2: 'single'}

education_options = {0: 'primary', 1: 'secondary', 2: 'tertiary', 3: 'unknown'}

housing_options = {0: 'no', 1: 'yes'}

default_options = {0: 'no', 1: 'yes'}

loan_options = {0: 'no', 1: 'yes'}

contact_options = {0: 'cellular', 1: 'telephone', 2: 'unknown'}

month_options = {0: 'apr', 1: 'aug', 2: 'dec', 3: 'feb', 4: 'jan', 5: 'jul',
                 6: 'jun', 7: 'mar', 8: 'may', 9: 'nov', 10: 'oct', 11: 'sep'}

poutcome_options = {0: 'failure', 1: 'other', 2: 'success', 3: 'unknown'}

# Get the selected values for each categorical variable
job_code = st.selectbox("JOB", options=list(job_options.keys()))
job = job_options.get(job_code)

marital_code = st.selectbox("MARITAL", options=list(marital_options.keys()))
marital = marital_options.get(marital_code)

education_code = st.selectbox("EDUCATION", options=list(education_options.keys()))
education = education_options.get(education_code)

housing_code = st.selectbox("HOUSING", options=list(housing_options.keys()))
housing = housing_options.get(housing_code)

default_code = st.selectbox("DEFAULT", options=list(default_options.keys()))
default = default_options.get(default_code)

loan_code = st.selectbox("LOAN", options=list(loan_options.keys()))
loan = loan_options.get(loan_code)

contact_code = st.selectbox("CONTACT", options=list(contact_options.keys()))
contact = contact_options.get(contact_code)

month_code = st.selectbox("MONTH", options=list(month_options.keys()))
month = month_options.get(month_code)

poutcome_code = st.selectbox("POUTCOME", options=list(poutcome_options.keys()))
poutcome = poutcome_options.get(poutcome_code)


client_data = [age, job, marital, education, default, balance, housing,loan, contact, day, month, duration, campaign, pdays,previous, poutcome]
data= np.array(list(client_data)).reshape(1,-1)

clf.predict(data)
if clf.predict(data)[0] == 1:
    st.write("The client subscribed a term deposit")
else:
    st.write("No subscribed a term deposit")
