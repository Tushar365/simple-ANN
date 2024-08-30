import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
from tensorflow.keras.models import load_model

# Load model
model = load_model('ann_reg_model.h5')

# Load the pickle files
with open('label_encoder_gender.pkl', "rb") as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_en.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Estimated salary Prediction')

# User input
options = {0: "No", 1: "Yes"}
CreditScore = st.number_input('Credit Score')
Geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
Gender = st.selectbox('Gender', label_encoder_gender.classes_)
Age = st.slider('Age', 18, 100)
Tenure = st.slider('Tenure', 0, 10)
Balance = st.number_input('Balance')
NumOfProducts = st.number_input('Number of Products')
HasCrCard = st.selectbox("Has Credit Card?", options.keys(), format_func=lambda x: options[x])
IsActiveMember = st.selectbox("Is Active Member?", options.keys(), format_func=lambda x: options[x])
Exited = st.selectbox("A churn customer?", options.keys(), format_func=lambda x: options[x])
# Predict button
if st.button('Predict'):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Geography': [Geography],
        'Gender': [label_encoder_gender.transform([Gender])[0]],# Encoding categorical Gender value
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'Exited' : [Exited]
    })

    # One hot encoding Geography
    geo_encoded = one_hot_encoder_geo.transform([[Geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

    # Concatenate the encoded columns with the input data
    input_data = pd.concat([input_data, geo_encoded_df], axis=1)
    input_data = input_data.drop(['Geography'], axis=1)


    # Scaling input data
    input_data = scaler.transform(input_data)

    # Predict churn
    result = model.predict(input_data)
    result_prob = result[0][0]
    st.write(f'estimated salary: {result_prob:.2f}')