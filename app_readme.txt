1.Importing Libraries:

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
from tensorflow.keras.models import load_model


'''
streamlit: A library for building interactive web apps in Python.
numpy, pandas: Standard libraries for numerical and data manipulation tasks.
tensorflow: A popular deep learning library used to load and use the ANN model.
sklearn.preprocessing: Modules to transform categorical data (LabelEncoder, OneHotEncoder) and scale numerical data (StandardScaler).
pickle: Used to load the pre-trained encoders and scaler.

'''
2. Loading the Pre-Trained Model and Encoders:

# Load model
model = load_model('annmodel.h5')

# Load the pickle files
with open('label_encoder_gender.pkl', "rb") as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_en.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

'''
load_model('annmodel.h5'): Loads the pre-trained ANN model saved in an HDF5 file.
Pickle files: These files contain pre-fitted LabelEncoder, OneHotEncoder, and StandardScaler objects. 
            These are used to transform the input data in the same way as it was done during training.

'''

3.Building the Streamlit App:

st.title('Customer Churn Prediction')
#This sets the title of the web app.

4. User Input Collection:

CreditScore = st.number_input('Credit Score')
Geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
Gender = st.selectbox('Gender', label_encoder_gender.classes_)
Age = st.slider('Age', 18, 100)
Tenure = st.slider('Tenure', 0, 10)
Balance = st.number_input('Balance')
NumOfProducts = st.number_input('Number of Products')
HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
IsActiveMember = st.selectbox("Is Active Member?", [0, 1])
EstimatedSalary = st.number_input('Estimated Salary')

'''
st.number_input(), st.selectbox(), st.slider(): These functions create various input fields (numeric, dropdown, slider) on the 
                                                Streamlit web app to collect user input.
'''


5. Predict Button Logic:


if st.button('Predict'):
#st.button('Predict'): Creates a "Predict" button that triggers the code within this block when clicked.
6. Preparing the Input Data:


# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Geography': [Geography],
    'Gender': [label_encoder_gender.transform([Gender])[0]], # Encoding categorical Gender value
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})
'''
This section creates a DataFrame from the user input, which will be used for making predictions.
label_encoder_gender.transform([Gender])[0]: Encodes the selected gender using the pre-fitted LabelEncoder.
'''

7. One-Hot Encoding for Geography:

# One hot encoding Geography
geo_encoded = one_hot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# Concatenate the encoded columns with the input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)
input_data = input_data.drop(['Geography'], axis=1)
'''
One-Hot Encoding: Converts the categorical Geography field into a set of binary columns.
get_feature_names_out(['Geography']): Ensures the column names are consistent with how the model was trained.
'''
8. Scaling the Input Data:

# Scaling input data
input_data = scaler.transform(input_data)

'''
scaler.transform(input_data): Applies the same scaling (normalization) that was applied to the training data.
                              This step ensures that the model receives data in the format it expects.
'''

9. Making the Prediction:

# Predict churn
result = model.predict(input_data)
result_prob = result[0][0]
st.write(f'Churn Probability: {result_prob:.2f}')

'''
model.predict(input_data): Uses the trained ANN model to predict the probability of churn based on the input data.
Displaying Results: The predicted probability is displayed on the app.

'''

10. Displaying the Churn Result:

if result_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
#This condition checks the predicted probability against a threshold of 0.5 and displays whether the customer is likely to churn.
Summary:
This Streamlit app takes user inputs related to customer data, processes these inputs (encoding, scaling),
and uses a pre-trained ANN model to predict the likelihood of customer churn.
The app provides an interactive interface where users can input data, click a button to get predictions, and 
see the results immediately.