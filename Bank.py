
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

# Load data function
def get_dataset(name):
    data = None
    if name == 'Bank':
        data = pd.read_csv('bank-full.csv')
        objList = data.select_dtypes(include = "object").columns
        le = LabelEncoder()
        for values in objList:
            data[values] = le.fit_transform(data[values].astype(str))
    elif name == 'Dummy':
        data = datasets.load_iris()  # Load dummy data for testing, replace with your actual dummy data
    return data

# Sidebar selections
dataset_name = st.sidebar.selectbox("Select Dataset", ("Bank", "Dummy"))

# Get dataset
data = get_dataset(dataset_name)

# Prepare X, y
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Classifier selection
classifier_name = st.sidebar.selectbox('Select classifier', ('GB',))

# Add parameter UI
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'GB':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

# Get classifier
def get_classifier(clf_name, params):
    if clf_name == 'GB':
        return GradientBoostingClassifier(n_estimators=params['n_estimators'],
                                          max_depth=params['max_depth'],
                                          random_state=1234)

# Train and evaluate classifier
clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Display accuracy
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

# Client data input
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

# Predict client data
client_data = np.array([age, balance, day, duration, campaign, pdays, previous])
client_data = np.append(client_data, [job, marital, education, housing, loan, contact, month, poutcome])
client_data = client_data.reshape(1, -1)

prediction = clf.predict(client_data)
if prediction == 1:
    st.write("The client subscribed a term deposit")
else:
    st.write("No subscribed a term deposit")
