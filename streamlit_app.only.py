import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
    color: #262730;
    background-color: #f0f2f6;
}
.stApp {
    max-width: 1600px;
    margin: 0 auto;
    padding: 30px;
}
h1 {
    color: #39A7FF;
    text-align: center;
    margin-bottom: 40px;
    font-size: 3em;
    font-weight: bold;
}
.stSidebar {
    background-color: #FFFFFF;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
}
.stSidebar h2 {
    color: #39A7FF;
    margin-bottom: 30px;
    font-size: 1.8em;
}
.stButton > button {
    background-color: #39A7FF;
    color: white;
    font-size: 18px;
    padding: 12px 24px;
    border-radius: 8px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
}
.stButton > button:hover {
    background-color: #1E86FF;
}
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data_from_github(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_data = response.content.decode('utf-8')
        return pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

github_url = "https://raw.githubusercontent.com/Uppala19/Airline-fare-Prediction/refs/heads/main/Updated_Flight_Fare_Data-_23_.csv"
data = load_data_from_github(github_url)

if data is not None:
    data['Total_Stops'] = data['Total_Stops'].replace("non-stop", 0)
    data['Total_Stops'] = data['Total_Stops'].fillna(0)

    def convert_stops_to_numeric(stops):
        if isinstance(stops, (int, float)):
            return stops
        elif isinstance(stops, str) and '→' in stops:
            return len(stops.split('→'))
        return 0

    data['Total_Stops'] = data['Total_Stops'].astype(str).apply(convert_stops_to_numeric)
    data['Total_Stops'] = pd.to_numeric(data['Total_Stops'], errors='coerce').fillna(0)

    data['Dep_Time'] = pd.to_datetime(data['Dep_Time'], errors='coerce')
    data['Dep_Time_hour'] = data['Dep_Time'].dt.hour
    data['Dep_Time_minute'] = data['Dep_Time'].dt.minute
    data.drop('Dep_Time', axis=1, inplace=True)

    data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'], errors='coerce')
    data['Arrival_Time_hour'] = data['Arrival_Time'].dt.hour
    data['Arrival_Time_minute'] = data['Arrival_Time'].dt.minute
    data.drop('Arrival_Time', axis=1, inplace=True)

    def convert_duration_to_minutes(duration):
        match = re.match(r'(\d+)h\s*(\d+)m', str(duration))
        if match:
            return int(match.group(1)) * 60 + int(match.group(2))
        match = re.match(r'(\d+)h', str(duration))
        if match:
            return int(match.group(1)) * 60
        match = re.match(r'(\d+)m', str(duration))
        if match:
            return int(match.group(1))
        return 0

    data['Duration_minutes'] = data['Duration'].apply(convert_duration_to_minutes)
    data.drop(['Duration', 'Additional_Info'], axis=1, inplace=True)

    data['Flight_Layover'] = data['Flight_Layover'].astype('category').cat.codes

    data['Booking_Date'] = pd.to_datetime(data['Booking_Date'], errors='coerce')
    data['Booking_Day'] = data['Booking_Date'].dt.day
    data['Booking_Month'] = data['Booking_Date'].dt.month
    data['Booking_Year'] = data['Booking_Date'].dt.year
    data.drop('Booking_Date', axis=1, inplace=True)

    data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], errors='coerce')
    data['Journey_Day'] = data['Date_of_Journey'].dt.day
    data['Journey_Month'] = data['Date_of_Journey'].dt.month

    data = data.dropna(axis=1, how='any')

    airline_mapping = dict(enumerate(data['Airline'].astype('category').cat.categories))
    source_mapping = dict(enumerate(data['Source'].astype('category').cat.categories))
    destination_mapping = dict(enumerate(data['Destination'].astype('category').cat.categories))
    cabin_class_mapping = dict(enumerate(data['Cabin_Class'].astype('category').cat.categories))

    model_data = data.copy()
    for col, mapping in zip(['Airline', 'Source', 'Destination', 'Cabin_Class'], [airline_mapping, source_mapping, destination_mapping, cabin_class_mapping]):
        model_data[col] = model_data[col].map(lambda x: list(mapping.keys())[list(mapping.values()).index(x)])

    for col in model_data.columns:
        model_data[col] = pd.to_numeric(model_data[col], errors='coerce')

    X = model_data.drop('Price', axis=1)
    y = model_data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    @st.cache_resource
    def train_model(X_train, y_train):
        model = RandomForestRegressor(random_state=100)
        model.fit(X_train, y_train)
        return model

    model = train_model(X_train, y_train)

    with st.sidebar:
        st.title("Flight Fare Prediction")
        st.markdown("Explore flight data and predict fares.")
        page = st.radio("Choose a section:", ["Model Evaluation", "Prediction"])

    st.title("✈️ Flight Fare Prediction App")

    if page == "Model Evaluation":
        st.header("Evaluate the Prediction Model")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.2f}")
        with col2:
            st.metric("R^2 Score", f"{r2:.2f}")

        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance.plot(kind='bar', ax=ax, color=sns.color_palette("Set2"))
        ax.set_title("Top 10 Features Based on Mutual Information")
        st.pyplot(fig)

    elif page == "Prediction":
        st.header("Predict Your Flight Fare")

        unique_airlines = data['Airline'].unique().tolist()
        unique_sources = data['Source'].unique().tolist()
        unique_destinations = data['Destination'].unique().tolist()
        unique_cabin_classes = data['Cabin_Class'].unique().tolist()

        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Source", options=unique_sources)
            destination = st.selectbox("Destination", options=unique_destinations)
            airline = st.selectbox("Airline", options=unique_airlines)
            stops = st.slider("Number of Stops", 0, 5, 0)
            cabin_class = st.selectbox("Cabin Class", options=unique_cabin_classes)

        with col2:
            journey_date = st.date_input("Journey Date", datetime.now())
            dep_hour = st.slider("Departure Hour", 0, 23, 12)
            dep_minute = st.slider("Departure Minute", 0, 59, 0)
            arrival_hour = st.slider("Arrival Hour", 0, 23, 15)
            arrival_minute = st.slider("Arrival Minute", 0, 59, 0)

        if st.button("Predict Fare"):
            input_data = pd.DataFrame({
                'Total_Stops': [stops],
                'Dep_Time_hour': [dep_hour],
                'Dep_Time_minute': [dep_minute],
                'Arrival_Time_hour': [arrival_hour],
                'Arrival_Time_minute': [arrival_minute],
                'Journey_Day': [journey_date.day],
                'Journey_Month': [journey_date.month],
                'Airline': [list(airline_mapping.keys())[list(airline_mapping.values()).index(airline)]],
                'Source': [list(source_mapping.keys())[list(source_mapping.values()).index(source)]],
                'Destination': [list(destination_mapping.keys())[list(destination_mapping.values()).index(destination)]],
                'Cabin_Class': [list(cabin_class_mapping.keys())[list(cabin_class_mapping.values()).index(cabin_class)]],
            })

            for col in X_train.columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[X_train.columns]

            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Flight Fare: ₹{prediction:.2f}")
else:
    st.error("Failed to load data. Check the GitHub URL and your internet connection.")
