
Project title:
Dynamic Airline Fare Prediction using Advanced Machine Learning Techniques.
Overview
This project uses advanced machine learning techniques to predict airline ticket prices based on historical and contextual data, enriched with external features like holidays, cabin class, and days until departure.A Streamlit web application was built for user-friendly, real-time fare prediction.
Dataset: The dataset was taken from kaggle, an open source .The data set consists of Indian domestic flights  in 2019.
https://www.kaggle.com/datasets/jillanisofttech/flight-price-prediction-dataset
This Scripts contains some additional features are added to already existing dataset.
Required libraries for this script:
•	pandas
•	numpy
•	scikit-learn
•	xgboost
•	matplotlib
•	seaborn
•	streamlit
•	requests
•	holidays
•	openpyxl

This script handles data preprocessing tasks such as missing value imputation, encoding categorical variables (e.g., airline, cabin class), and feature engineering. It also includes steps to convert timestamps and calculate additional features like days until departure and flight duration.
This Script contains graphs such as scatter plot, box plot, bar graphs, heat map, line graph, and some other graphs to show the relationship between features.
Model Evaluation:
Models in the script we used are:
•	Linear Regression
•	Decision Tree Regressor
•	Random Forest Regressor
•	Gradient Boosting Regressor
•	XGBoost Regressor
•	Hypertuned Random Forest

Performance metrics we are used in this scripts.
   R², MAE, MSE, RMSE, MAPE
The best model in the script is found to be the Hypertune random forest model with  R² of ~ 0.89 and MAE=951.90
Streamlit Web app is built based on basic HTML,CSS with simple python language 
Clone the repository:
 git clone https://github.com/Uppala19/Airline-fare-Prediction.gitcd Airline-fare-Prediction
web app link:https://airline-fare-prediction-erreezvsvkamsajztpjlej.streamlit.app/.

