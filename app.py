import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from tensorflow.keras.models import load_model


MODELS_DIR = "models"

BANK_MODEL_PATH = os.path.join(MODELS_DIR, "bank_churn_ann.keras")
BANK_SCALER_PATH = os.path.join(MODELS_DIR, "bank_scaler.pkl")

TELCO_MODEL_PATH = os.path.join(MODELS_DIR, "telco_churn_ann.keras")
TELCO_SCALER_PATH = os.path.join(MODELS_DIR, "telco_scaler.pkl")

ECOMM_MODEL_PATH = os.path.join(MODELS_DIR, "ecomm_churn_ann.keras")
ECOMM_SCALER_PATH = os.path.join(MODELS_DIR, "ecomm_scaler.pkl")


GEOGRAPHY_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP = {"Female": 0, "Male": 1}

DOMAINS = {
    "bank": "Bank Customers (Churn_Modelling.csv)",
    "telco": "Telco Customers (WA_Fn-UseC_-Telco-Customer-Churn.csv)",
    "ecomm": "E-Commerce Customers",
}


def load_artifacts():
    artifacts: dict[str, dict] = {}

    if os.path.exists(BANK_MODEL_PATH) and os.path.exists(BANK_SCALER_PATH):
        bank_model = load_model(BANK_MODEL_PATH)
        bank_scaler = joblib.load(BANK_SCALER_PATH)
        artifacts["bank"] = {"model": bank_model, "scaler": bank_scaler}

    if os.path.exists(TELCO_MODEL_PATH) and os.path.exists(TELCO_SCALER_PATH):
        telco_model = load_model(TELCO_MODEL_PATH)
        telco_bundle = joblib.load(TELCO_SCALER_PATH)
        artifacts["telco"] = {
            "model": telco_model,
            "scaler": telco_bundle["scaler"],
            "columns": telco_bundle["columns"],
        }

    if os.path.exists(ECOMM_MODEL_PATH) and os.path.exists(ECOMM_SCALER_PATH):
        ecomm_model = load_model(ECOMM_MODEL_PATH)
        ecomm_bundle = joblib.load(ECOMM_SCALER_PATH)
        artifacts["ecomm"] = {
            "model": ecomm_model,
            "scaler": ecomm_bundle["scaler"],
            "columns": ecomm_bundle["columns"],
        }

    if not artifacts:
        raise RuntimeError(
            "No trained models were found. "
            "Run train_model.py first to generate models for each dataset."
        )

    return artifacts


ARTIFACTS = load_artifacts()

app = Flask(__name__)


@app.context_processor
def inject_common_context():
    # Make sure templates always see `domains` and a default `domain`
    return {"domains": DOMAINS, "domain": "bank"}


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        geography_options=list(GEOGRAPHY_MAP.keys()),
        gender_options=list(GENDER_MAP.keys()),
    )


@app.route("/predict", methods=["POST"])
def predict():
    domain = request.form.get("domain", "bank")

    if domain == "ecomm" and "ecomm" in ARTIFACTS:
        return _predict_ecomm()

    if domain == "telco" and "telco" in ARTIFACTS:
        return _predict_telco()

    # Default to bank model
    return _predict_bank()


def _predict_bank():
    try:
        geography = request.form.get("geography")
        gender = request.form.get("gender")
        credit_score = float(request.form.get("credit_score", "0"))
        age = int(request.form.get("age", "0"))
        tenure = int(request.form.get("tenure", "0"))
        balance = float(request.form.get("balance", "0"))
        num_products = int(request.form.get("num_products", "1"))
        has_cr_card = int(request.form.get("has_cr_card", "0"))
        is_active_member = int(request.form.get("is_active_member", "0"))
        estimated_salary = float(request.form.get("estimated_salary", "0"))
    except ValueError:
        return redirect(url_for("index"))

    geo_encoded = GEOGRAPHY_MAP.get(geography, 0)
    gender_encoded = GENDER_MAP.get(gender, 0)

    features = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geo_encoded,
                "Gender": gender_encoded,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active_member,
                "EstimatedSalary": estimated_salary,
            }
        ]
    )

    bank_model = ARTIFACTS["bank"]["model"]
    bank_scaler = ARTIFACTS["bank"]["scaler"]

    features_scaled = bank_scaler.transform(features)

    prob = float(bank_model.predict(features_scaled)[0][0])
    will_churn = prob > 0.5

    return render_template(
        "result.html",
        probability=f"{prob * 100:.2f}",
        will_churn=will_churn,
        domain="bank",
        domains=DOMAINS,
    )


def _predict_telco():
    # For telco, we mirror the preprocessing used in train_model.py
    try:
        gender = request.form.get("gender")
        senior_citizen = int(request.form.get("SeniorCitizen", "0"))
        partner = request.form.get("Partner")
        dependents = request.form.get("Dependents")
        tenure = int(request.form.get("tenure", "0"))
        phone_service = request.form.get("PhoneService")
        multiple_lines = request.form.get("MultipleLines")
        internet_service = request.form.get("InternetService")
        online_security = request.form.get("OnlineSecurity")
        online_backup = request.form.get("OnlineBackup")
        device_protection = request.form.get("DeviceProtection")
        tech_support = request.form.get("TechSupport")
        streaming_tv = request.form.get("StreamingTV")
        streaming_movies = request.form.get("StreamingMovies")
        contract = request.form.get("Contract")
        paperless_billing = request.form.get("PaperlessBilling")
        payment_method = request.form.get("PaymentMethod")
        monthly_charges = float(request.form.get("MonthlyCharges", "0"))
        total_charges = float(request.form.get("TotalCharges", "0"))
    except ValueError:
        return redirect(url_for("telco"))

    row = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    df = pd.DataFrame([row])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    X_encoded = pd.get_dummies(df, drop_first=True)

    telco_artifacts = ARTIFACTS["telco"]
    telco_scaler = telco_artifacts["scaler"]
    telco_columns = telco_artifacts["columns"]

    # Align columns with training-time columns
    X_encoded = X_encoded.reindex(columns=telco_columns, fill_value=0)

    X_scaled = telco_scaler.transform(X_encoded)

    telco_model = telco_artifacts["model"]
    prob = float(telco_model.predict(X_scaled)[0][0])
    will_churn = prob > 0.5

    return render_template(
        "result.html",
        probability=f"{prob * 100:.2f}",
        will_churn=will_churn,
        domain="telco",
        domains=DOMAINS,
    )


def _predict_ecomm():
    # Mirror preprocessing used in train_model.load_and_preprocess_ecomm
    try:
        tenure = float(request.form.get("Tenure", "0"))
        preferred_login_device = request.form.get("PreferredLoginDevice")
        city_tier = int(request.form.get("CityTier", "1"))
        warehouse_to_home = float(request.form.get("WarehouseToHome", "0"))
        preferred_payment_mode = request.form.get("PreferredPaymentMode")
        gender = request.form.get("Gender")
        hour_spend_on_app = float(request.form.get("HourSpendOnApp", "0"))
        num_devices = int(request.form.get("NumberOfDeviceRegistered", "1"))
        prefered_order_cat = request.form.get("PreferedOrderCat")
        satisfaction_score = int(request.form.get("SatisfactionScore", "3"))
        marital_status = request.form.get("MaritalStatus")
        num_address = int(request.form.get("NumberOfAddress", "1"))
        complain = int(request.form.get("Complain", "0"))
        order_amount_hike = float(
            request.form.get("OrderAmountHikeFromlastYear", "0")
        )
        coupon_used = int(request.form.get("CouponUsed", "0"))
        order_count = int(request.form.get("OrderCount", "0"))
        days_since_last_order = int(request.form.get("DaySinceLastOrder", "0"))
        cashback_amount = float(request.form.get("CashbackAmount", "0"))
    except ValueError:
        return redirect("/ecomm")

    row = {
        "Tenure": tenure,
        "PreferredLoginDevice": preferred_login_device,
        "CityTier": city_tier,
        "WarehouseToHome": warehouse_to_home,
        "PreferredPaymentMode": preferred_payment_mode,
        "Gender": gender,
        "HourSpendOnApp": hour_spend_on_app,
        "NumberOfDeviceRegistered": num_devices,
        "PreferedOrderCat": prefered_order_cat,
        "SatisfactionScore": satisfaction_score,
        "MaritalStatus": marital_status,
        "NumberOfAddress": num_address,
        "Complain": complain,
        "OrderAmountHikeFromlastYear": order_amount_hike,
        "CouponUsed": coupon_used,
        "OrderCount": order_count,
        "DaySinceLastOrder": days_since_last_order,
        "CashbackAmount": cashback_amount,
    }

    df = pd.DataFrame([row])

    # Handle any missing numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    X_encoded = pd.get_dummies(df, drop_first=True)

    ecomm_artifacts = ARTIFACTS["ecomm"]
    ecomm_scaler = ecomm_artifacts["scaler"]
    ecomm_columns = ecomm_artifacts["columns"]

    X_encoded = X_encoded.reindex(columns=ecomm_columns, fill_value=0)

    X_scaled = ecomm_scaler.transform(X_encoded)

    ecomm_model = ecomm_artifacts["model"]
    prob = float(ecomm_model.predict(X_scaled)[0][0])
    will_churn = prob > 0.5

    return render_template(
        "result.html",
        probability=f"{prob * 100:.2f}",
        will_churn=will_churn,
        domain="ecomm",
        domains=DOMAINS,
    )


@app.route("/telco", methods=["GET"])
def telco():
    if "telco" not in ARTIFACTS:
        # If telco model is not trained, send user back to bank form with a simple message.
        return redirect(url_for("index"))

    return render_template(
        "telco.html",
        domain="telco",
        domains=DOMAINS,
    )


@app.route("/ecomm", methods=["GET"])
def ecomm():
    if "ecomm" not in ARTIFACTS:
        return redirect(url_for("index"))

    return render_template(
        "ecomm.html",
        domain="ecomm",
        domains=DOMAINS,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

