from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("churn_model_v2.pkl")

class Customer(BaseModel):
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

    gender_Male: int
    Partner_Yes: int
    Dependents_Yes: int
    PhoneService_Yes: int

    MultipleLines_No_phone_service: int
    MultipleLines_Yes: int

    InternetService_Fiber_optic: int
    InternetService_No: int

    OnlineSecurity_No_internet_service: int
    OnlineSecurity_Yes: int

    OnlineBackup_No_internet_service: int
    OnlineBackup_Yes: int

    DeviceProtection_No_internet_service: int
    DeviceProtection_Yes: int

    TechSupport_No_internet_service: int
    TechSupport_Yes: int

    StreamingTV_No_internet_service: int
    StreamingTV_Yes: int

    StreamingMovies_No_internet_service: int
    StreamingMovies_Yes: int

    Contract_One_year: int
    Contract_Two_year: int

    PaperlessBilling_Yes: int

    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int


@app.post("/predict")
def predict(customer: Customer):

    data = pd.DataFrame([customer.dict()])

    data.rename(columns={
        "MultipleLines_No_phone_service": "MultipleLines_No phone service",
        "InternetService_Fiber_optic": "InternetService_Fiber optic",
        "OnlineSecurity_No_internet_service": "OnlineSecurity_No internet service",
        "OnlineBackup_No_internet_service": "OnlineBackup_No internet service",
        "DeviceProtection_No_internet_service": "DeviceProtection_No internet service",
        "TechSupport_No_internet_service": "TechSupport_No internet service",
        "StreamingTV_No_internet_service": "StreamingTV_No internet service",
        "StreamingMovies_No_internet_service": "StreamingMovies_No internet service",
        "Contract_One_year": "Contract_One year",
        "Contract_Two_year": "Contract_Two year",
        "PaymentMethod_Credit_card_automatic": "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check"
    }, inplace=True)

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]

    result = (
        "Likely to Churn"
        if prediction[0] == 1
        else "Not Likely to Churn"
    )

    return {
        "prediction": result,
        "churn_probability": round(float(probability), 2)
    }