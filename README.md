# telecom-customer-churn
# Customer Churn Prediction System

## Overview

This project predicts whether a telecom customer is likely to churn using Machine Learning.

A Random Forest Classifier was trained on customer subscription data and deployed using FastAPI for real-time predictions.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* FastAPI
* Joblib

## Model Performance

* Accuracy: 78.92%
* Precision: 64.29%
* Recall: 45.84%
* F1 Score: 53.52%

## Features

* Customer churn prediction
* Probability scoring
* REST API using FastAPI
* Interactive Swagger documentation

## API Example

Response:

{
"prediction": "Not Likely to Churn",
"churn_probability": 0.33
}

## Future Improvements

* Streamlit dashboard
* Cloud deployment
* Model explainability
* Improved feature engineering
