import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import xgboost as xgb



def app():
    st.title('Bank Marketing in Streamlit')


    dataset_name = st.sidebar.selectbox(
        "Select Dataset", ("Bank Marketing",""))

    st.write(f"## {dataset_name} Dataset")

    classifier_name = st.sidebar.selectbox(
        'Select classifier', ('Random Forest','SVM','GB','XGBoost','MLP','KNN','LR')
    )

    def get_dataset(name):
        data = None
        name == 'Bank Marketing'
        data = pd.read_csv('bank-full.csv', delimiter=";")
        objList = data.select_dtypes(include="object").columns
        le = LabelEncoder()
        for values in objList:
            data[values] = le.fit_transform(data[values].astype(str))
            data = data
        X = data.loc[:, data.columns != 'y']
        y = data.loc[:, data.columns == 'y'].values.ravel()
        return X, y

    X, y = get_dataset(dataset_name)
    st.write('Shape of dataset:', X.shape)
    st.write('number of classes:', len(np.unique(y)))

    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == 'SVM':
            C = st.sidebar.slider('C (SVM)', 0.01, 10.0)
            kernel = st.sidebar.selectbox(
                'Kernel (SVM)', ['linear', 'poly', 'rbf', 'sigmoid'])
            params['C'] = C
            params['kernel'] = kernel
        elif clf_name == 'XGBoost':
            n_estimators = st.sidebar.slider('n_estimators (XGBoost)', 1, 100, 10)
            learning_rate = st.sidebar.slider('learning_rate (XGBoost)', 0.01, 1.0, 0.1)
            max_depth = st.sidebar.slider('max_depth (XGBoost)', 2, 15)
            params['n_estimators'] = n_estimators
            params['learning_rate'] = learning_rate
            params['max_depth'] = max_depth
        elif clf_name == 'GB':
            max_depth = st.sidebar.slider('max_depth (GB)', 2, 15)
            n_estimators = st.sidebar.slider('n_estimators (GB)', 1, 100)
            learning_rate = st.sidebar.slider(
                'learning_rate (GB)', 0.01, 1.0, 0.1)
            params['max_depth'] = max_depth
            params['n_estimators'] = n_estimators
            params['learning_rate'] = learning_rate
        elif clf_name == 'Random Forest':
            max_depth = st.sidebar.slider(
                'max_depth (Random Forest)', 2, 15)
            n_estimators = st.sidebar.slider(
                'n_estimators (Random Forest)', 1, 1000)
            params['max_depth'] = max_depth
            params['n_estimators'] = n_estimators
        elif clf_name == 'MLP':
            hidden_layer_sizes_1 = st.sidebar.slider(
                 'Hidden Layer Sizes 1 (MLP)', 24, 256)
            hidden_layer_sizes_2 = st.sidebar.slider(
                'Hidden Layer Sizes 2 (MLP)', 24, 256)
            hidden_layer_sizes_3 = st.sidebar.slider(
                'Hidden Layer Sizes 3 (MLP)', 24, 256)
            activation = st.sidebar.selectbox(
                'Activation Function (MLP)', ['identity', 'logistic', 'tanh', 'relu'])
            solver = st.sidebar.selectbox(
                'Solver (MLP)', ['lbfgs', 'sgd', 'adam'])
            params['hidden_layer_sizes_1'] = hidden_layer_sizes_1
            params['hidden_layer_sizes_2'] = hidden_layer_sizes_2
            params['hidden_layer_sizes_3'] = hidden_layer_sizes_3
            params['activation'] = activation
            params['solver'] = solver
        elif clf_name == 'KNN':
            n_neighbors = st.sidebar.slider('Number of Neighbors (KNN)', 1, 30, key='knn_n_neighbors_slider')
            params['n_neighbors'] = n_neighbors
        elif clf_name == 'Logistic Regression':
            pass
        return params

    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'SVM':
            clf = SVC(C=params['C'], kernel=params['kernel'])
        elif clf_name == 'XGBoost':
            clf = xgb.XGBClassifier(n_estimators=params['n_estimators'], 
                                    learning_rate=params['learning_rate'], 
                                    max_depth=params['max_depth'], 
                                    random_state=1234)
        elif clf_name == 'GB':
            clf = GradientBoostingClassifier(
                n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                learning_rate=params['learning_rate'], random_state=1234)
        elif clf_name == 'Random Forest':
            clf = RandomForestClassifier(
                n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                random_state=1234)
        elif clf_name == 'MLP':
            hidden_layer_sizes = (params['hidden_layer_sizes_1'], params['hidden_layer_sizes_2'], params['hidden_layer_sizes_3'])
            clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=params['activation'],
                                solver=params['solver'], random_state=42)
        elif clf_name == 'KNN':
            n_neighbors = st.sidebar.slider('Number of Neighbors (KNN)', 1, 30)
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif clf_name == 'Logistic Regression':
            clf = LogisticRegression()
        return clf

    # Obtain parameters
    params = add_parameter_ui(classifier_name)

    # Get classifier using the obtained parameters
    clf = get_classifier(classifier_name, params)

    st.write(f'Classifier = {classifier_name}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.write(f'Accuracy =', acc)

    st.write('Enter the values for prediction')
    age = st.number_input("AGE",value=None, placeholder="Type a number...")
    balance = st.number_input("BALANCE",value=None, placeholder="Type a number...")
    day = st.number_input("DAY",value=None, placeholder="Type a number...")
    duration = st.number_input("DURATION",value=None, placeholder="Type a number...")
    campaign = st.number_input("CAMPAIGN",value=None, placeholder="Type a number...")
    pdays = st.number_input("PDAYS",value=None, placeholder="Type a number...")
    previous = st.number_input("PREVIOUS",value=None, placeholder="Type a number...")

    # Mapping of textual options to numeric codes for each categorical variable
    job_options = {'management': 1, 'technician': 2, 'entrepreneur': 3, 'blue-collar': 4,
                   'unknown': 5, 'retired': 6, 'admin.': 7, 'services': 8, 'self-employed': 9,
                   'unemployed': 10, 'housemaid': 11, 'student': 12}

    marital_options = {'married': 1, 'single': 2, 'divorced': 3}

    education_options = {'tertiary': 1, 'secondary': 2,
                         'unknown': 3, 'primary': 4}

    housing_options = {'yes': 1, 'no': 0}

    default_options = {'yes': 1, 'no': 0}

    loan_options = {'no': 0, 'yes': 1}

    contact_options = {'unknown': 0, 'cellular': 1, 'telephone': 2}

    month_options = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

    poutcome_options = {'failure': 0, 'other': 1,
                        'success': 2, 'unknown': 3}

    # Display dropdown menus with categorical options
    job_code = st.selectbox("JOB", options=list(job_options.keys()))
    marital_code = st.selectbox(
        "MARITAL", options=list(marital_options.keys()))
    education_code = st.selectbox(
        "EDUCATION", options=list(education_options.keys()))
    housing_code = st.selectbox(
        "HOUSING", options=list(housing_options.keys()))
    default_code = st.selectbox(
        "DEFAULT", options=list(default_options.keys()))
    loan_code = st.selectbox("LOAN", options=list(loan_options.keys()))
    contact_code = st.selectbox(
        "CONTACT", options=list(contact_options.keys()))
    month_code = st.selectbox("MONTH", options=list(month_options.keys()))
    poutcome_code = st.selectbox(
        "POUTCOME", options=list(poutcome_options.keys()))


    # Add a button to trigger the model prediction
    if st.button("Run Model"):
        # Get numerical values corresponding to the selected categorical options
        job = job_options.get(job_code)
        marital = marital_options.get(marital_code)
        education = education_options.get(education_code)
        housing = housing_options.get(housing_code)
        default = default_options.get(default_code)
        loan = loan_options.get(loan_code)
        contact = contact_options.get(contact_code)
        month = month_options.get(month_code)
        poutcome = poutcome_options.get(poutcome_code)

        # Prepare the input data for prediction
        client_data = [age, job, marital, education, default, balance, housing,loan, contact, day, month, duration, campaign, pdays,previous, poutcome]
        data= np.array(list(client_data)).reshape(1,-1)

        # Make prediction
        prediction = clf.predict(data)

        # Display prediction result
        if prediction[0] == 1:
            st.write("The client subscribed to a term deposit.")
        else:
            st.write("The client did not subscribe to a term deposit.")
            # Obtain predicted probabilities
    y_probs = clf.predict_proba(X_test)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    st.write('Receiver Operating Characteristic (ROC) Curve')
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    st.pyplot(fig)
  
