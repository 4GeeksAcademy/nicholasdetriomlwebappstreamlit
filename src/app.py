import joblib
import streamlit as st
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("trained_model.joblib")

model = load_model()
#print(type(model))

@st.cache_data
def preprocess_input(val1, val2, val3, val4, val5):
    # Create a dictionary from the input values
    input_dict = {
    "school_setting": [val1],
    "school_type": [val2],
    "teaching_method": [val3],
    "gender": [val4],
    "lunch": [val5]
    }
    # print(input_dict)
    
    # Convert dictionary to DataFrame
    input_df = pd.DataFrame(input_dict)
        
    return input_df

st.title("Student Test Scores Predictor")

val1 = st.selectbox(
    'school_setting', 
    ('Urban', 'Suburban', 'Rural')
)
val2 = st.selectbox(
    'school_type', 
    ('Non-public', 'Public')
)
val3 = st.selectbox(
    'teaching_method', 
    ('Standard', 'Experimental')
)
val4 = st.selectbox(
    'gender', 
    ('Female', 'Male')
)
val5 = st.selectbox(
    'lunch', 
    ('Does not qualify', 'Qualifies for reduced/free lunch')
)
# print("val1:", val1)
# print("val2:", val2)
# print("val3:", val3)
# print("val4:", val4)
# print("val5:", val5)

if st.button("Predict"):
    # Preprocess the input data
    data_processed = preprocess_input(val1, val2, val3, val4, val5)

    prediction = str(model.predict(data_processed)[0])
    # print("Prediction:", prediction)
    # pred_class = class_dict[prediction]
    st.write("Prediction:", prediction)

