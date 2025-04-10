import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# GitHub URL for the dataset
github_url = "https://raw.githubusercontent.com/hemanth18-web/dp-ml/refs/heads/main/Updated_Flight_Fare_Data%20(20).csv"

# Function to load the dataset from GitHub
@st.cache_data
def load_data_from_github(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_data = response.content.decode('utf-8')
        data = pd.read_csv(io.StringIO(csv_data))
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download the dataset from GitHub: {e}")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Failed to parse the CSV data: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Load the dataset
data = load_data_from_github(github_url)

# --- STREAMLIT APP ---
st.title("Flight Fare Data Exploration and Prediction")

if data is not None:
    # --- Data Cleaning and Conversion ---
    # Replace "non-stop" with 0
    data['Total_Stops'] = data['Total_Stops'].replace("non-stop", 0)

    # Handle missing values (NaN)
    data['Total_Stops'] = data['Total_Stops'].replace('NaN', np.nan)  # Replace string 'NaN' with actual NaN
    data['Total_Stops'] = data['Total_Stops'].fillna(0)  # Fill NaN with 0 (or another appropriate value)

    # --- Identify Non-Convertible Values ---
    def is_number(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    non_numeric_values = data['Total_Stops'][data['Total_Stops'].apply(lambda x: not is_number(x))]
    unique_non_numeric = non_numeric_values.unique()

    if len(unique_non_numeric) > 0:
        st.warning(f"Found non-numeric values in 'Total_Stops': {unique_non_numeric}")
        # Replace these values with a suitable numeric value (e.g., 0) or NaN
        # For example, if you want to replace all non-numeric values with 0:
        for val in unique_non_numeric:
            data['Total_Stops'] = data['Total_Stops'].replace(val, 0)
    else:
        st.success("No non-numeric values found in 'Total_Stops'")

    # Convert 'Total_Stops' to numeric
    data['Total_Stops'] = pd.to_numeric(data['Total_Stops'])

    # Convert Date_of_Journey to datetime
    data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'])

    # Extract features: day, month
    data['Journey_Day'] = data['Date_of_Journey'].dt.day
    data['Journey_Month'] = data['Date_of_Journey'].dt.month

    # Convert Dep_Time and Arrival_Time to datetime objects and extract features
    data['Dep_Time'] = pd.to_datetime(data['Dep_Time'], errors='coerce')
    data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'], errors='coerce')

    data['Dep_Time_Hour'] = data['Dep_Time'].dt.hour
    data['Dep_Time_Minute'] = data['Dep_Time'].dt.minute
    data['Arrival_Time_Hour'] = data['Arrival_Time'].dt.hour
    data['Arrival_Time_Minute'] = data['Arrival_Time'].dt.minute

    st.header("Data Preview")
    st.dataframe(data.head())

    st.header("Data Summary")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")

    # --- VISUALIZATIONS ---
    st.header("Visualizations")

    # Airline Distribution
    st.subheader("Airline Distribution")
    fig_airline, ax_airline = plt.subplots(figsize=(10, 6))
    sns.countplot(x="Airline", data=data, ax=ax_airline, palette="muted", order=data['Airline'].value_counts().index)  # Order by frequency
    ax_airline.tick_params(axis='x', rotation=90)
    st.pyplot(fig_airline)

    # Price vs. Number of Stops (Scatter Plot)
    st.subheader("Price vs. Number of Stops")
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="Price", y="Total_Stops", data=data, ax=ax_scatter)
    st.pyplot(fig_scatter)

    # Price Distribution (Histogram)
    st.subheader("Price Distribution")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Price'], kde=True, ax=ax_hist)
    st.pyplot(fig_hist)

    # --- ADDED: Scatter plot with customized aesthetics ---
    st.header("Price vs Number of Stops (Customized)")
    fig_scatter_custom, ax_scatter_custom = plt.subplots(figsize=(10, 6))
    scatter = ax_scatter_custom.scatter(data['Price'], data['Total_Stops'], s=80, alpha=0.7, c=data['Price'], cmap='viridis', edgecolors='black')
    ax_scatter_custom.set_title('Price vs Number of Stops', fontsize=14, fontweight='bold')
    ax_scatter_custom.set_xlabel('Price', fontsize=12)
    ax_scatter_custom.set_ylabel('Number of Stops', fontsize=12)
    ax_scatter_custom.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    fig_scatter_custom.colorbar(scatter, label='Price')  # Add colorbar to the figure
    st.pyplot(fig_scatter_custom)

    # --- ADDED: Airline Distribution Countplot with Styling ---
    st.header("Airline Distribution (Styled)")
    fig_countplot, ax_countplot = plt.subplots(figsize=(10, 6))
    sns.countplot(x="Airline", data=data, hue="Airline", palette="muted", legend=False, ax=ax_countplot)
    ax_countplot.set_title("✈️ Airline Distribution ✈️", fontweight="bold", fontsize=14, color="#80aaff")
    ax_countplot.set_xlabel("Airline")
    ax_countplot.set_ylabel("Count")
    ax_countplot.tick_params(axis='x', rotation=90, labelsize=8)
    fig_countplot.tight_layout()
    st.pyplot(fig_countplot)

    # Create the line plot
    st.subheader("Ticket Price Trends Over Time")
    fig_lineplot, ax_lineplot = plt.subplots(figsize=(10, 6))
    sns.lineplot(x="Date_of_Journey", y="Price", data=data, hue="Airline", marker="o", palette="viridis", ax=ax_lineplot)  # Changed x to "Date_of_Journey" and y to "Price"
    ax_lineplot.set_title("Ticket Price Trends Over Time", fontsize=14, fontweight="bold")
    ax_lineplot.set_xlabel("Date", fontsize=12)
    ax_lineplot.set_ylabel("Price (₹) ", fontsize=12)
    ax_lineplot.tick_params(axis='x', rotation=90)
    fig_lineplot.tight_layout()
    st.pyplot(fig_lineplot)

    # Assuming 'Days_Until_Departure' is in your data, we first calculate the frequency of each unique value
    st.subheader("Days Until Departure (Line Plot)")
    days_count = data['Days_Until_Departure'].value_counts().sort_index()
    fig_days_line, ax_days_line = plt.subplots(figsize=(10, 6))
    ax_days_line.plot(days_count.index, days_count.values, marker='o', color='skyblue', linewidth=2)
    ax_days_line.set_title('Days Until Departure (Line Plot)', fontsize=12, fontweight='bold')
    ax_days_line.set_xlabel('Days Until Departure', fontsize=12)
    ax_days_line.set_ylabel('Count', fontsize=12)
    fig_days_line.tight_layout()
    st.pyplot(fig_days_line)

    # Set style
    st.subheader("Price Distribution by Cabin Class (Customized)")
    sns.set(style="whitegrid")
    fig_boxplot_cabin, ax_boxplot_cabin = plt.subplots(figsize=(10, 8))
    sns.boxplot(x="Cabin_Class", y="Price", data=data, palette="Set1", hue="Airline", ax=ax_boxplot_cabin)
    ax_boxplot_cabin.set_title("Price Distribution by Cabin Class (Customized)", fontsize=14, fontweight='bold')
    ax_boxplot_cabin.set_xlabel("Cabin Class", fontsize=12)
    ax_boxplot_cabin.set_ylabel("Price ()", fontsize=12)
    ax_boxplot_cabin.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    fig_boxplot_cabin.tight_layout()
    st.pyplot(fig_boxplot_cabin)

    # Set style for a clean, modern look
    st.subheader("Price Distribution by Cabin Class")
    sns.set(style="whitegrid")
    fig_boxplot_cabin2, ax_boxplot_cabin2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Cabin_Class", y="Price", data=data, palette="Set2", hue="Cabin_Class", ax=ax_boxplot_cabin2)
    ax_boxplot_cabin2.set_title("Price Distribution by Cabin Class ", fontsize=14, fontweight='bold', color="#2c3e50")
    ax_boxplot_cabin2.set_xlabel("Cabin Class", fontsize=12)
    ax_boxplot_cabin2.set_ylabel("Price ($)", fontsize=12)
    ax_boxplot_cabin2.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)
    fig_boxplot_cabin2.tight_layout()
    st.pyplot(fig_boxplot_cabin2)

    # Set style for a clean, modern look
    st.subheader("Price Distribution by Airline")
    sns.set(style="whitegrid")
    fig_boxplot_airline, ax_boxplot_airline = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="Airline", y="Price", data=data.sort_values('Price', ascending=False), hue="Airline", palette="Set2", ax=ax_boxplot_airline)
    ax_boxplot_airline.set_title(" Price Distribution by Airline", fontsize=14, fontweight='bold', color="#2c3e50")
    ax_boxplot_airline.set_xlabel("Airline ", fontsize=12)
    ax_boxplot_airline.set_ylabel("Price", fontsize=12)
    ax_boxplot_airline.tick_params(axis='x', rotation=90)
    ax_boxplot_airline.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    fig_boxplot_airline.tight_layout()
    st.pyplot(fig_boxplot_airline)

    # --- Feature Engineering and Preprocessing for Model ---
    # Use original string values for 'Airline', 'Source', 'Destination', 'Cabin_Class'
    # No Label Encoding here

    # Print column names for debugging
    print("Column Names Before Feature Selection:")
    print(data.columns)

    # Prepare data for model training
    # Ensure all required columns exist.  If 'Duration_Hours' or 'Duration_Minutes' are missing, calculate them.
    if 'Duration_Hours' not in data.columns or 'Duration_Minutes' not in data.columns:
        data['Duration'] = (data['Arrival_Time'] - data['Dep_Time']).dt.total_seconds() / 60  # Duration in minutes
        data['Duration_Hours'] = data['Duration'] // 60
        data['Duration_Minutes'] = data['Duration'] % 60

    # Select features, handling potential missing columns
    features = ['Total_Stops', 'Dep_Time_Hour', 'Dep_Time_Minute',
                'Arrival_Time_Hour', 'Arrival_Time_Minute', 'Duration_Hours', 'Duration_Minutes',
                'Journey_Day', 'Journey_Month', 'Days_Until_Departure', 'Airline', 'Source', 'Destination', 'Cabin_Class']

    # Check if all features exist in the DataFrame
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        st.error(f"Missing features: {missing_features}.  Please check your data.")
        st.stop()  # Stop execution if features are missing

    # One-Hot Encode Categorical Features
    data = pd.get_dummies(data, columns=['Airline', 'Source', 'Destination', 'Cabin_Class'], drop_first=True)

    # Update the features list to include the one-hot encoded columns
    features = [col for col in data.columns if col not in ['Price', 'Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration']]  # Exclude non-feature columns

    X = data[features]
    y = data['Price']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Print data types and missing values for debugging ---
    print("Data types of X_train columns:")
    print(X_train.dtypes)
    print("Number of missing values in X_train:")
    print(X_train.isnull().sum())

    # --- Handle missing values (if any) ---
    X_train = X_train.fillna(X_train.mean())  # Impute missing values with the mean
    X_test = X_test.fillna(X_test.mean())

    # --- Model Training ---
    st.header("Random Forest Model Training")
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train, y_train)

    # --- Model Evaluation ---
    y_pred = random_forest_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R^2 Score: {r2:.2f}")

    # --- Feature Importance Plot ---
    st.subheader("Feature Importance")
    feature_importance = pd.Series(random_forest_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_feature_importance, ax_feature_importance = plt.subplots(figsize=(10, 6))
    feature_importance.plot(kind='bar', ax=ax_feature_importance)
    ax_feature_importance.set_title("Feature Importance from Random Forest")
    ax_feature_importance.set_ylabel("Importance Score")
    st.pyplot(fig_feature_importance)

    # --- Prediction Form ---
    st.header("Flight Price Prediction")

    # Get unique values for selectboxes
    # Extract base column names before one-hot encoding
    unique_airlines = data.columns[data.columns.str.startswith('Airline_')].str.replace('Airline_', '').tolist()
    unique_sources = data.columns[data.columns.str.startswith('Source_')].str.replace('Source_', '').tolist()
    unique_destinations = data.columns[data.columns.str.startswith('Destination_')].str.replace('Destination_', '').tolist()
    unique_cabin_classes = data.columns[data.columns.str.startswith('Cabin_Class_')].str.replace('Cabin_Class_', '').tolist()

    # User input fields
    source = st.selectbox("Source", options=unique_sources)
    destination = st.selectbox("Destination", options=unique_destinations)
    stops = st.slider("Number of Stops", min_value=0, max_value=5, value=0)
    airline = st.selectbox("Airline", options=unique_airlines)
    dep_hour = st.slider("Departure Hour", min_value=0, max_value=23, value=12)
    dep_minute = st.slider("Departure Minute", min_value=0, max_value=59, value=0)
    arrival_hour = st.slider("Arrival Hour", min_value=0, max_value=23, value=14)
    arrival_minute = st.slider("Arrival Minute", min_value=0, max_value=59, value=0)
    duration_hours = st.slider("Duration Hours", min_value=0, max_value=24, value=2)
    duration_minutes = st.slider("Duration Minutes", min_value=0, max_value=59, value=0)
    journey_day = st.slider("Journey Day", min_value=1, max_value=31, value=15)
    journey_month = st.slider("Journey Month", min_value=1, max_value=12, value=3)
    cabin_class = st.selectbox("Cabin Class", options=unique_cabin_classes)
    days_until_departure = st.slider("Days Until Departure", min_value=1, max_value=365, value=30)

    # Prediction button
    if st.button("Predict Price"):
        # Prepare input data
        input_data = {}
        for feature in features:
            input_data[feature] = [0]  # Initialize all features to 0

        # Set the values for the user-selected features
        input_data[f'Airline_{airline}'] = [1]  # Set the selected airline to 1
        input_data[f'Source_{source}'] = [1]  # Set the selected source to 1
        input_data[f'Destination_{destination}'] = [1]  # Set the selected destination to 1
        input_data[f'Cabin_Class_{cabin_class}'] = [1]  # Set the selected cabin class to 1
        input_data['Total_Stops'] = [stops]
        input_data['Dep_Time_Hour'] = [dep_hour]
        input_data['Dep_Time_Minute'] = [dep_minute]
        input_data['Arrival_Time_Hour'] = [arrival_hour]
        input_data['Arrival_Time_Minute'] = [arrival_minute]
        input_data['Duration_Hours'] = [duration_hours]
        input_data['Duration_Minutes'] = [duration_minutes]
        input_data['Journey_Day'] = [journey_day]
        input_data['Journey_Month'] = [journey_month]
        input_data['Days_Until_Departure'] = [days_until_departure]

        input_df = pd.DataFrame(input_data)

        # Make prediction
        prediction = random_forest_model.predict(input_df)
        st.success(f"Predicted Flight Price: ₹{prediction[0]:.2f}")

else:
    st.write("Failed to load data.  Check the GitHub URL and your internet connection.")
