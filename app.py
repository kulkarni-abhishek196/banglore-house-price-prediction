import streamlit as st
import pandas as pd
import joblib
import numpy as np

df = pd.read_csv('/Users/apple/Developer/ML/bangalore-house-price-prediction/datasets/dataset.csv')



st.title("Begaluru house price prediction")
st.write("Insert your inputs")

available_options = ["Ready To Move", "Not Ready To Move"]
availability = st.selectbox(
    label="Select availability",
    options=available_options,
    index=0
)

unique_locations = df['location'].str.rstrip().dropna().drop_duplicates()

sorted_locations = sorted(unique_locations)

selected_location = st.selectbox(
    label="select the location",
    options=sorted_locations,
    index=0
)

total_sqft = st.number_input("Enter total sqft")

bath = ["1", "2", "3", "4"]
bathroom = st.selectbox(
    label="enter number of bathrooms",
    options=bath,
    index=0
)

bal = ["1", "2", "3", "4"]
balcony = st.selectbox(
    label="enter number of balcony",
    options=bal,
    index=0
)

size_input = ["1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK"]
size = st.selectbox(
    label="choose your house size",
    options=size_input,
    index=0
)


bhkNumbers = size.split(' ')
sqft_per_BHK = total_sqft / int(bhkNumbers[0])

model = joblib.load('/Users/apple/Developer/ML/bangalore-house-price-prediction/models/house_prediction_model.pkl')
if st.button("predict"):
    input_data = pd.DataFrame([{
        "availability": availability,
        "location": selected_location,
        "total_sqft": total_sqft,
        "bath": bathroom,
        "balcony": balcony,
        "size_filled": size,
        "total_sqft_log": np.log1p(total_sqft),
        "sqft_perBHK": sqft_per_BHK
    }])

    price_pred_log = model.predict(input_data)
    price_pred_ruppes = np.expm1(price_pred_log)

    st.write("Your predicted house price is :", price_pred_ruppes)


