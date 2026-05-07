import os
from functools import wraps
 
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for
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
    return {"domains": DOMAINS, "domain": "bank"}
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  AUTHENTICATION
#  Set CHURN_API_KEY in your environment before running.
#  Windows:  set CHURN_API_KEY=your-secret-key
#  Default "dev-secret-key" is fine for local testing.
# ═══════════════════════════════════════════════════════════════════════════════
 
API_KEY = os.environ.get("CHURN_API_KEY", "dev-secret-key")
 
 
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if key != API_KEY:
            return jsonify({"error": "Unauthorized – send a valid X-API-Key header"}), 401
        return f(*args, **kwargs)
    return decorated
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  WEB ROUTES  (your original form-based interface – unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
 
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
        order_amount_hike = float(request.form.get("OrderAmountHikeFromlastYear", "0"))
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
        return redirect(url_for("index"))
    return render_template("telco.html", domain="telco", domains=DOMAINS)
 
 
@app.route("/ecomm", methods=["GET"])
def ecomm():
    if "ecomm" not in ARTIFACTS:
        return redirect(url_for("index"))
    return render_template("ecomm.html", domain="ecomm", domains=DOMAINS)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  REST API ROUTES  (new JSON-based interface)
# ═══════════════════════════════════════════════════════════════════════════════
 
def _make_result(prob: float, domain: str) -> dict:
    will_churn = prob > 0.5
    return {
        "domain":                domain,
        "churn_probability":     round(prob, 4),
        "churn_probability_pct": f"{prob * 100:.2f}%",
        "risk":                  "High" if will_churn else "Low",
        "will_churn":            will_churn,
    }
 
 
# ── GET /api/health  (no auth needed) ────────────────────────────────────────
# curl http://localhost:5000/api/health
 
@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status":            "ok",
        "loaded_domains":    list(ARTIFACTS.keys()),
        "available_domains": list(DOMAINS.keys()),
    })
 
 
# ── POST /api/predict  (requires X-API-Key header) ───────────────────────────
# curl -X POST http://localhost:5000/api/predict \
#   -H "Content-Type: application/json" \
#   -H "X-API-Key: dev-secret-key" \
#   -d '{"domain":"bank","CreditScore":650,"Geography":"France","Gender":"Female",
#        "Age":35,"Tenure":5,"Balance":75000,"NumOfProducts":2,
#        "HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":55000}'
 
@app.route("/api/predict", methods=["POST"])
@require_api_key
def api_predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400
 
    domain = data.get("domain", "bank").lower()
    if domain not in ARTIFACTS:
        return jsonify({
            "error": f"Domain '{domain}' model not loaded. Available: {list(ARTIFACTS.keys())}"
        }), 422
 
    try:
        runners = {"bank": _run_bank, "telco": _run_telco, "ecomm": _run_ecomm}
        prob = runners[domain](data)
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 422
 
    return jsonify(_make_result(prob, domain))
 
 
# ── POST /api/predict/batch  (requires X-API-Key header, max 100 records) ────
# curl -X POST http://localhost:5000/api/predict/batch \
#   -H "Content-Type: application/json" \
#   -H "X-API-Key: dev-secret-key" \
#   -d '{"domain":"bank","customers":[{...},{...}]}'
 
@app.route("/api/predict/batch", methods=["POST"])
@require_api_key
def api_predict_batch():
    data = request.get_json(force=True, silent=True)
    if not data or "customers" not in data:
        return jsonify({"error": "Provide JSON with 'domain' and 'customers' list"}), 400
 
    domain = data.get("domain", "bank").lower()
    customers = data["customers"]
 
    if not isinstance(customers, list) or len(customers) == 0:
        return jsonify({"error": "'customers' must be a non-empty list"}), 422
    if len(customers) > 100:
        return jsonify({"error": "Batch size limit is 100 per request"}), 422
    if domain not in ARTIFACTS:
        return jsonify({
            "error": f"Domain '{domain}' model not loaded. Available: {list(ARTIFACTS.keys())}"
        }), 422
 
    runner = {"bank": _run_bank, "telco": _run_telco, "ecomm": _run_ecomm}[domain]
    results = []
    for idx, customer in enumerate(customers):
        try:
            prob = runner(customer)
            results.append({"index": idx, **_make_result(prob, domain)})
        except Exception as exc:
            results.append({"index": idx, "error": str(exc)})
 
    return jsonify({"domain": domain, "total": len(results), "results": results})
 
 
# ── Inference helpers (mirror your form handlers but accept a plain dict) ─────
 
def _run_bank(data: dict) -> float:
    required = ["CreditScore", "Geography", "Gender", "Age", "Tenure",
                "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
                "EstimatedSalary"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing fields for bank domain: {missing}")
 
    geo = GEOGRAPHY_MAP.get(str(data["Geography"]).capitalize())
    gender = GENDER_MAP.get(str(data["Gender"]).capitalize())
    if geo is None:
        raise ValueError(f"Geography must be one of {list(GEOGRAPHY_MAP)}")
    if gender is None:
        raise ValueError(f"Gender must be one of {list(GENDER_MAP)}")
 
    features = pd.DataFrame([{
        "CreditScore":     float(data["CreditScore"]),
        "Geography":       geo,
        "Gender":          gender,
        "Age":             float(data["Age"]),
        "Tenure":          float(data["Tenure"]),
        "Balance":         float(data["Balance"]),
        "NumOfProducts":   int(data["NumOfProducts"]),
        "HasCrCard":       int(data["HasCrCard"]),
        "IsActiveMember":  int(data["IsActiveMember"]),
        "EstimatedSalary": float(data["EstimatedSalary"]),
    }])
    scaled = ARTIFACTS["bank"]["scaler"].transform(features)
    return float(ARTIFACTS["bank"]["model"].predict(scaled)[0][0])
 
 
def _run_telco(data: dict) -> float:
    row = {
        "gender":           data.get("gender"),
        "SeniorCitizen":    int(data.get("SeniorCitizen", 0)),
        "Partner":          data.get("Partner"),
        "Dependents":       data.get("Dependents"),
        "tenure":           int(data.get("tenure", 0)),
        "PhoneService":     data.get("PhoneService"),
        "MultipleLines":    data.get("MultipleLines"),
        "InternetService":  data.get("InternetService"),
        "OnlineSecurity":   data.get("OnlineSecurity"),
        "OnlineBackup":     data.get("OnlineBackup"),
        "DeviceProtection": data.get("DeviceProtection"),
        "TechSupport":      data.get("TechSupport"),
        "StreamingTV":      data.get("StreamingTV"),
        "StreamingMovies":  data.get("StreamingMovies"),
        "Contract":         data.get("Contract"),
        "PaperlessBilling": data.get("PaperlessBilling"),
        "PaymentMethod":    data.get("PaymentMethod"),
        "MonthlyCharges":   float(data.get("MonthlyCharges", 0)),
        "TotalCharges":     float(data.get("TotalCharges", 0)),
    }
    df = pd.DataFrame([row])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
 
    X_encoded = pd.get_dummies(df, drop_first=True)
    telco_artifacts = ARTIFACTS["telco"]
    X_encoded = X_encoded.reindex(columns=telco_artifacts["columns"], fill_value=0)
    X_scaled = telco_artifacts["scaler"].transform(X_encoded)
    return float(telco_artifacts["model"].predict(X_scaled)[0][0])
 
 
def _run_ecomm(data: dict) -> float:
    row = {
        "Tenure":                     float(data.get("Tenure", 0)),
        "PreferredLoginDevice":        data.get("PreferredLoginDevice"),
        "CityTier":                    int(data.get("CityTier", 1)),
        "WarehouseToHome":             float(data.get("WarehouseToHome", 0)),
        "PreferredPaymentMode":        data.get("PreferredPaymentMode"),
        "Gender":                      data.get("Gender"),
        "HourSpendOnApp":              float(data.get("HourSpendOnApp", 0)),
        "NumberOfDeviceRegistered":    int(data.get("NumberOfDeviceRegistered", 1)),
        "PreferedOrderCat":            data.get("PreferedOrderCat"),
        "SatisfactionScore":           int(data.get("SatisfactionScore", 3)),
        "MaritalStatus":               data.get("MaritalStatus"),
        "NumberOfAddress":             int(data.get("NumberOfAddress", 1)),
        "Complain":                    int(data.get("Complain", 0)),
        "OrderAmountHikeFromlastYear": float(data.get("OrderAmountHikeFromlastYear", 0)),
        "CouponUsed":                  int(data.get("CouponUsed", 0)),
        "OrderCount":                  int(data.get("OrderCount", 0)),
        "DaySinceLastOrder":           int(data.get("DaySinceLastOrder", 0)),
        "CashbackAmount":              float(data.get("CashbackAmount", 0)),
    }
    df = pd.DataFrame([row])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
 
    X_encoded = pd.get_dummies(df, drop_first=True)
    ecomm_artifacts = ARTIFACTS["ecomm"]
    X_encoded = X_encoded.reindex(columns=ecomm_artifacts["columns"], fill_value=0)
    X_scaled = ecomm_artifacts["scaler"].transform(X_encoded)
    return float(ecomm_artifacts["model"].predict(X_scaled)[0][0])
 
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
 
