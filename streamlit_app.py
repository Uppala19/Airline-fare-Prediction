import streamlit as st
import pandas as pd
import numpy as np
import re  # Import the regular expression module
import requests
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# --- STREAMLIT APP ---
st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* General styling */
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
    /* Header styling */
    h1 {
        color: #39A7FF;
        text-align: center;
        margin-bottom: 40px;
        font-size: 3em;
        font-weight: bold;
    }
    /* Input elements styling */
    .stSelectbox, .stSlider, .stDateInput {
        margin-bottom: 25px;
    }
    /* Button styling */
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
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #e1e1e8;
        border-radius: 8px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.08);
    }
    /* Metric styling */
    .stMetric {
        background-color: white;
        border-radius: 8px;
        padding: 25px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
    }
    .stMetric label {
        font-weight: bold;
        color: #555;
        font-size: 1.2em;
    }
    .stMetric > div:nth-child(2) {
        font-size: 2.2em;
        color: #39A7FF;
    }
    /* Visualization styling */
    .stPlot {
        background-color: white;
        border-radius: 8px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
    }
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background-color: #e1e1e8;
        margin: 30px 0;
    }
    /* Info boxes */
    .info-box {
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .info-box h3 {
        color: #39A7FF;
        font-size: 1.4em;
        margin-bottom: 10px;
    }
    .info-box p {
        color: #555;
        font-size: 1.1em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# GitHub URL for the dataset
github_url = "https://raw.githubusercontent.com/Uppala19/Airline-fare-Prediction/refs/heads/main/updated_dataset.csv"


# Function to load the dataset from GitHub
@st.cache_data
def load_data_from_github(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        csv_data = response.content.decode("utf-8")
        data = pd.read_csv(io.StringIO(csv_data))
        return data
    except requests.exceptions.HTTPError as e:
        st.error(
            f"HTTPError: Could not download the dataset from GitHub. Status code: {e.response.status_code}"
        )
        return None
    except requests.exceptions.ConnectionError as e:
        st.error(
            "ConnectionError: Could not connect to GitHub. Please check your internet connection."
        )
        return None
    except requests.exceptions.Timeout as e:
        st.error("TimeoutError: Request to GitHub timed out.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"RequestException: An error occurred while making the request to GitHub: {e}")
        return None
    except pd.errors.ParserError as e:
        st.error(f"ParserError: Failed to parse the CSV data: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# Load the dataset
data = load_data_from_github(github_url)

# --- DATA PREPROCESSING (Move this outside the conditional page logic) ---
if data is not None:
    # --- Data Cleaning and Conversion ---
    data["Total_Stops"] = data["Total_Stops"].replace("non-stop", 0)
    data["Total_Stops"] = data["Total_Stops"].replace("NaN", np.nan)
    data["Total_Stops"] = data["Total_Stops"].fillna(0)

    def convert_stops_to_numeric(stops):
        if isinstance(stops, (int, float)):
            return stops
        elif isinstance(stops, str) and "→" in stops:
            return len(stops.split("→"))
        else:
            return 0

    data["Total_Stops"] = data["Total_Stops"].astype(str).apply(convert_stops_to_numeric)
    data["Total_Stops"] = pd.to_numeric(data["Total_Stops"], errors="coerce").fillna(0)
    try:
        data["Dep_Time"] = pd.to_datetime(data["Dep_Time"], errors="coerce")
        data["Dep_Time_hour"] = data["Dep_Time"].dt.hour
        data["Dep_Time_minute"] = data["Dep_Time"].dt.minute
        data.drop("Dep_Time", axis=1, inplace=True, errors="ignore")
    except Exception as e:
        st.error(f"Error processing 'Dep_Time' column: {e}")
    try:
        data["Arrival_Time"] = pd.to_datetime(data["Arrival_Time"], errors="coerce")
        data["Arrival_Time_hour"] = data["Arrival_Time"].dt.hour
        data["Arrival_Time_minute"] = data["Arrival_Time"].dt.minute
        data.drop("Arrival_Time", axis=1, inplace=True, errors="ignore")
    except Exception as e:
        st.error(f"Error processing 'Arrival_Time' column: {e}")

    def convert_duration_to_minutes(duration):
        try:
            match = re.match(r"(\d+)h\s*(\d+)m", str(duration))
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                return hours * 60 + minutes
            else:
                match_hours = re.match(r"(\d+)h", str(duration))
                match_minutes = re.match(r"(\d+)m", str(duration))
                if match_hours:
                    hours = int(match_hours.group(1))
                    return hours * 60
                elif match_minutes:
                    minutes = int(match_minutes.group(1))
                    return minutes
                else:
                    return 0
        except:
            return 0

    data["Duration_minutes"] = data["Duration"].apply(convert_duration_to_minutes)
    data.drop("Duration", axis=1, inplace=True, errors="ignore")
    data.drop("Additional_Info", axis=1, inplace=True, errors="ignore")
    data["Flight_Layover"] = data["Flight_Layover"].astype("category").cat.codes
    try:
        data["Booking_Date"] = pd.to_datetime(data["Booking_Date"], errors="coerce")
        data["Booking_Day"] = data["Booking_Date"].dt.day
        data["Booking_Month"] = data["Booking_Date"].dt.month
        data["Booking_Year"] = data["Booking_Date"].dt.year
        data.drop("Booking_Date", axis=1, inplace=True, errors="ignore")
    except Exception as e:
        st.error(f"Error processing 'Booking_Date' column: {e}")
    data["Date_of_Journey"] = pd.to_datetime(data["Date_of_Journey"], errors="coerce")
    data["Journey_Day"] = data["Date_of_Journey"].dt.day
    data["Journey_Month"] = data["Date_of_Journey"].dt.month
    # Remove columns with any NaN values
    data = data.dropna(axis=1, how="any")
    # Create mappings for encoding
    airline_mapping = dict(enumerate(data["Airline"].astype("category").cat.categories))
    source_mapping = dict(enumerate(data["Source"].astype("category").cat.categories))
    destination_mapping = dict(enumerate(data["Destination"].astype("category").cat.categories))
    # cabin_class_mapping = dict(enumerate(data["Cabin_Class"].astype("category").cat.categories)) # Remove cabin class mapping
    # Create a copy of the data for the model, where we'll encode
    model_data = data.copy()
    # Encode the categorical columns in the model data
    for col, mapping in zip(
        ["Airline", "Source", "Destination"],  # Removed Cabin_Class
        [airline_mapping, source_mapping, destination_mapping],  # Removed cabin_class_mapping
    ):
        model_data[col] = model_data[col].map(
            lambda x: list(mapping.keys())[list(mapping.values()).index(x)]
        )
    # Attempt to convert all columns to numeric
    for col in model_data.columns:
        try:
            model_data[col] = pd.to_numeric(model_data[col])
        except ValueError:
            st.error(f"Could not convert column '{col}' to numeric.  Please investigate.")
            st.stop()

    # Drop Cabin_Class from model_data if it exists
    if 'Cabin_Class' in model_data.columns:
        model_data.drop('Cabin_Class', axis=1, inplace=True, errors='ignore')

    X = model_data.drop(["Price"], axis=1, errors="ignore")
    y = model_data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    @st.cache_resource  # Use cache_resource for models
    def train_model(X_train, y_train):
        model = RandomForestRegressor(random_state=100)
        param_dist = {
            "n_estimators": [150, 200, 250],
            "max_depth": [10, 20, 30],
            "min_samples_split": [2, 5, 10],
        }
        rf = RandomForestRegressor(random_state=100)
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=5,  # Try 5 random combinations (not 3, adjusted for better coverage)
            cv=3,  # 3-fold cross-validation
            scoring="r2",  # Use 'accuracy' for classification
            random_state=100,  # Reproducibility
            n_jobs=-1,  # Use all CPU cores
            verbose=1,  # Show progress
        )
        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        return model

    random_forest_model = train_model(X_train, y_train)  # Train the model

    # --- Main App Content ---
    st.title("✈️ Flight Fare Prediction App")

    st.header("Predict Your Flight Fare")
    st.markdown("Enter your flight details below to get an estimated fare.")

    # Get unique values for selectboxes
    unique_airlines = data["Airline"].unique().tolist()
    unique_sources = data["Source"].unique().tolist()
    unique_destinations = data["Destination"].unique().tolist()
    # unique_cabin_classes = data["Cabin_Class"].unique().tolist() # Remove cabin class

    # Input fields using columns for layout
    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Source", options=unique_sources, help="Select the origin city")
        destination = st.selectbox(
            "Destination", options=unique_destinations, help="Select the destination city"
        )
        airline = st.selectbox("Airline", options=unique_airlines, help="Select the airline")
        stops = st.slider("Number of Stops", min_value=0, max_value=5, value=0, help="Number of layovers")
        # cabin_class = st.selectbox("Cabin Class", options=unique_cabin_classes, help="Select the cabin class") # Remove cabin class

    with col2:
        # Use date_input for journey date
        journey_date = st.date_input("Journey Date", datetime.now(), help="Select the date of travel")
        dep_hour = st.slider("Departure Hour", min_value=0, max_value=23, value=12, help="Hour of departure")
        dep_minute = st.slider("Departure Minute", min_value=0, max_value=59, value=0, help="Minute of departure")
        arrival_hour = st.slider("Arrival Hour", min_value=0, max_value=23, value=15, help="Hour of arrival")
        arrival_minute = st.slider("Arrival Minute", min_value=0, max_value=59, value=0, help="Minute of arrival")

    st.markdown("<hr>", unsafe_allow_html=True)  # Visual divider

    if st.button("Predict Fare"):
        # Prepare input data
        input_data = pd.DataFrame(
            {
                "Total_Stops": [stops],
                "Dep_Time_hour": [dep_hour],
                "Dep_Time_minute": [dep_minute],
                "Arrival_Time_hour": [arrival_hour],
                "Arrival_Time_minute": [arrival_minute],
                "Journey_Day": [journey_date.day],
                "Journey_Month": [journey_date.month],
                "Airline": [list(airline_mapping.keys())[list(airline_mapping.values()).index(airline)]],
                "Source": [list(source_mapping.keys())[list(source_mapping.values()).index(source)]],
                "Destination": [
                    list(destination_mapping.keys())[list(destination_mapping.values()).index(destination)]
                ],
                # "Cabin_Class": [
                #     list(cabin_class_mapping.keys())[list(cabin_class_mapping.values()).index(cabin_class)]
                # ], # Remove cabin class
            }
        )

        # Ensure all columns from training data are present in input data
        for col in X_train.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Remove Cabin_Class from input_data if it exists
        if 'Cabin_Class' in input_data.columns:
            input_data.drop('Cabin_Class', axis=1, inplace=True, errors='ignore')

        input_data = input_data[X_train.columns]

        # Make prediction
        predicted_fare = random_forest_model.predict(input_data)[0]
        st.success(f"Predicted Flight Fare: ₹{predicted_fare:.2f}")

else:
    st.error("Failed to load data. Check the GitHub URL and your internet connection.")
