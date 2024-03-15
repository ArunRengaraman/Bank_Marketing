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

# Mapping of textual options to numeric codes for each categorical variable
job_options = {'management': 1, 'technician': 2, 'entrepreneur': 3, 'blue-collar': 4,
               'unknown': 5, 'retired': 6, 'admin.': 7, 'services': 8, 'self-employed': 9,
               'unemployed': 10, 'housemaid': 11, 'student': 12}

marital_options = {'married': 1, 'single': 2, 'divorced': 3}

education_options = {'tertiary': 1, 'secondary': 2, 'unknown': 3, 'primary': 4}

housing_options = {'yes': 1, 'no': 0}

default_options = {'yes': 1, 'no': 0}

loan_options = {'no': 0, 'yes': 1}

contact_options = {'unknown': 0, 'cellular': 1, 'telephone': 2}

month_options = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

poutcome_options = {'failure': 0, 'other': 1, 'success': 2, 'unknown': 3}

# Display dropdown menus with categorical options
job = st.selectbox("JOB", options=list(job_options.keys()))
marital = st.selectbox("MARITAL", options=list(marital_options.keys()))
education = st.selectbox("EDUCATION", options=list(education_options.keys()))
housing = st.selectbox("HOUSING", options=list(housing_options.keys()))
default = st.selectbox("DEFAULT", options=list(default_options.keys()))
loan = st.selectbox("LOAN", options=list(loan_options.keys()))
contact = st.selectbox("CONTACT", options=list(contact_options.keys()))
month = st.selectbox("MONTH", options=list(month_options.keys()))
poutcome = st.selectbox("POUTCOME", options=list(poutcome_options.keys()))

# Get numerical values corresponding to the selected categorical options
job_code = job_options.get(job)
marital_code = marital_options.get(marital)
education_code = education_options.get(education)
housing_code = housing_options.get(housing)
default_code = default_options.get(default)
loan_code = loan_options.get(loan)
contact_code = contact_options.get(contact)
month_code = month_options.get(month)
poutcome_code = poutcome_options.get(poutcome)




client_data = [age, job, marital, education, default, balance, housing,loan, contact, day, month, duration, campaign, pdays,previous, poutcome]
data= np.array(list(client_data)).reshape(1,-1)

clf.predict(data)
if clf.predict(data)[0] == 1:
    st.write("The client subscribed a term deposit")
else:
    st.write("No subscribed a term deposit")
