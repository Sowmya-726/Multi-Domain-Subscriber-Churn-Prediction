import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


MODELS_DIR = "models"

# Bank dataset (Churn_Modelling.csv)
BANK_DATA_PATH = os.path.join("data", "Churn_Modelling.csv")
BANK_MODEL_PATH = os.path.join(MODELS_DIR, "bank_churn_ann.keras")
BANK_SCALER_PATH = os.path.join(MODELS_DIR, "bank_scaler.pkl")

# Telco dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv)
TELCO_DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
TELCO_MODEL_PATH = os.path.join(MODELS_DIR, "telco_churn_ann.keras")
TELCO_SCALER_PATH = os.path.join(MODELS_DIR, "telco_scaler.pkl")

# E-commerce dataset (E comm.csv)
ECOMM_DATA_PATH = os.path.join("data", "E comm.csv")
ECOMM_MODEL_PATH = os.path.join(MODELS_DIR, "ecomm_churn_ann.keras")
ECOMM_SCALER_PATH = os.path.join(MODELS_DIR, "ecomm_scaler.pkl")


GEOGRAPHY_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP = {"Female": 0, "Male": 1}


def load_and_preprocess_bank(csv_path: str):
    df = pd.read_csv(csv_path)

    # Drop identifier-like columns that do not help prediction
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    # Encode categorical features using fixed mappings
    df["Geography"] = df["Geography"].map(GEOGRAPHY_MAP)
    df["Gender"] = df["Gender"].map(GENDER_MAP)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def load_and_preprocess_telco(csv_path: str):
    df = pd.read_csv(csv_path)

    # Target: churn yes/no -> 1/0
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # Clean and drop identifier
    df = df.drop(["customerID", "Churn"], axis=1)

    # Convert TotalCharges to numeric and handle missing
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # One-hot encode categorical variables
    X = pd.get_dummies(df, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Keep feature columns for use at inference time
    feature_columns = X.columns.tolist()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns


def load_and_preprocess_ecomm(csv_path: str):
    df = pd.read_csv(csv_path)

    # Target is already 0/1
    y = df["Churn"]

    # Drop identifier and target from features
    df = df.drop(["CustomerID", "Churn"], axis=1)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # One-hot encode categorical variables
    X = pd.get_dummies(df, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_columns = X.columns.tolist()

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns


def build_model(input_dim: int) -> Sequential:
    model = Sequential()
    model.add(Dense(6, activation="relu", input_shape=(input_dim,)))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_save_bank():
    if not os.path.exists(BANK_DATA_PATH):
        raise FileNotFoundError(
            f"Could not find bank dataset at {BANK_DATA_PATH}. "
            f"Place Churn_Modelling.csv inside a 'data' folder in your project."
        )

    os.makedirs(MODELS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_bank(BANK_DATA_PATH)

    model = build_model(input_dim=X_train.shape[1])

    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Bank dataset – confusion matrix:")
    print(cm)
    print(f"Bank dataset – accuracy: {acc:.4f}")

    model.save(BANK_MODEL_PATH)
    joblib.dump(scaler, BANK_SCALER_PATH)
    print(f"Saved bank model to {BANK_MODEL_PATH}")
    print(f"Saved bank scaler to {BANK_SCALER_PATH}")


def train_and_save_telco():
    if not os.path.exists(TELCO_DATA_PATH):
        print(
            f"Telco dataset not found at {TELCO_DATA_PATH}. "
            "Skipping telco model training."
        )
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_columns,
    ) = load_and_preprocess_telco(TELCO_DATA_PATH)

    model = build_model(input_dim=X_train.shape[1])

    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Telco dataset – confusion matrix:")
    print(cm)
    print(f"Telco dataset – accuracy: {acc:.4f}")

    model.save(TELCO_MODEL_PATH)

    # Store scaler and feature columns together for inference
    joblib.dump({"scaler": scaler, "columns": feature_columns}, TELCO_SCALER_PATH)
    print(f"Saved telco model to {TELCO_MODEL_PATH}")
    print(f"Saved telco scaler and columns to {TELCO_SCALER_PATH}")


def train_and_save_ecomm():
    if not os.path.exists(ECOMM_DATA_PATH):
        print(
            f"E-commerce dataset not found at {ECOMM_DATA_PATH}. "
            "Skipping e-commerce model training."
        )
        return

    os.makedirs(MODELS_DIR, exist_ok=True)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        feature_columns,
    ) = load_and_preprocess_ecomm(ECOMM_DATA_PATH)

    model = build_model(input_dim=X_train.shape[1])

    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("E-commerce dataset – confusion matrix:")
    print(cm)
    print(f"E-commerce dataset – accuracy: {acc:.4f}")

    model.save(ECOMM_MODEL_PATH)
    joblib.dump({"scaler": scaler, "columns": feature_columns}, ECOMM_SCALER_PATH)
    print(f"Saved e-commerce model to {ECOMM_MODEL_PATH}")
    print(f"Saved e-commerce scaler and columns to {ECOMM_SCALER_PATH}")


if __name__ == "__main__":
    train_and_save_bank()
    train_and_save_telco()
    train_and_save_ecomm()

