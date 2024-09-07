# model deployment using streamlit

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib, json, os, sys
from sklearn import set_config
set_config(transform_output='pandas')

# Load the filepaths
FILEPATHS_FILE = 'config/filepaths.json'
with open(FILEPATHS_FILE) as f:
    FPATHS = json.load(f)
    
# Define the load train or test data function with caching
# @st.cache_data
# def load_Xy_data(fpath):
#     return joblib.load(fpath)
    
# @st.cache_resource
# def load_model_ml(fpath):
#     return joblib.load(fpath)


@st.cache_data
def load_Xy_data(fpath):
    train_path = fpath['data']['ml']['train']
    X_train, y_train =  joblib.load(train_path)
    test_path = fpath['data']['ml']['test']
    X_test, y_test = joblib.load(test_path)
    return X_train, y_train, X_test, y_test
 
@st.cache_resource
def load_model_ml(fpath):
    model_path = fpath['models']['linear_regression']
    linreg = joblib.load(model_path)
    return linreg

@st.cache_resource
def get_explainer(_model_pipe, X_train, labels):
    X_train_sc = _model_pipe[0].transform(X_train)
    feature_names = _model_pipe[0].get_feature_names_out()
    explainer = LimeTabularExplainer(
                    X_train_sc,
                    mode='classification',
                    feature_names=feature_names,
                    class_names=labels,
                    random_state=42
                    )
    return explainer

@st.cache_resource
def explain_instance(_explainer, _model_pipe, instance_to_explain):
    instance_to_explain_sc = _model_pipe[0].transform(instance_to_explain)
    explanation = _explainer.explain_instance(instance_to_explain_sc[0],
                                             _model_pipe[-1].predict_proba
                                             )
    return explanation
    
### Start of App
st.title('House Prices Prediction')
st.image('images/house.jpeg')
st.sidebar.header("House Features")
# # Include the banner image
# st.image(FPATHS['images']['banner'])


# Load training data
# X_train, y_train = load_Xy_data(fpath=FPATHS['data']['ml']['train'])
# # Load testing data
# X_test, y_test = load_Xy_data(fpath=FPATHS['data']['ml']['test'])

X_train, y_train, X_test, y_test = load_Xy_data(FPATHS)
# # Load model
# linreg = load_model_ml(fpath = FPATHS['models']['linear_regression'])

linreg = load_model_ml(FPATHS)


# # Create widgets for each feature
#bedrooms
Bedrooms = st.sidebar.slider('Bedrooms',
                            min_value = X_train['bedrooms'].min(),
                            max_value = X_train['bedrooms'].max(),
                            step = 1, value = 3)

#bathrooms
Bathrooms = st.sidebar.slider('Bathrooms',
                             min_value = X_train['bathrooms'].min(),
                             max_value = X_train['bathrooms'].max(),
                             step = .25, value = 2.5)

#sqft_living
sqft_living = st.sidebar.number_input('Sqft Living Area',
                                     min_value=290,
                                     max_value=X_train['sqft_living'].max(),
                                     step=150, value=2500)


# Add text for entering features
st.subheader("Select values using the sidebar on the left.\n Then check the box below to predict the price.")

st.sidebar.subheader("Enter/select House Features For Prediction")

# Define function to convert widget values to dataframe

def get_X_to_predict():
    X_to_predict = pd.DataFrame({'bedrooms': Bedrooms,
                                 'bathrooms':Bathrooms,
                                 'sqft_living':sqft_living},
                             index=['House'])
    return X_to_predict

def get_prediction(model,X_to_predict):
    return  model.predict(X_to_predict)[0]
    
if st.sidebar.button('Predict'):
    
    X_to_pred = get_X_to_predict()
    new_pred = get_prediction(linreg, X_to_pred)
    
    st.markdown(f"> #### Model Predicted Price = ${new_pred:,.0f}")
    
else:
    st.empty()

