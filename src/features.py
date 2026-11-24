import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INTERNET_ADDON = ["No", "Yes", "No internet service"]
ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Categorical inputs and the options a user can pick in the web form
CHOICES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    **{col: INTERNET_ADDON for col in ADDON_COLS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

NUMERIC_INPUTS = ["tenure", "MonthlyCharges", "TotalCharges"]
NUM_COLS = NUMERIC_INPUTS + ["num_addons", "avg_monthly_spend"]
CAT_COLS = list(CHOICES.keys())
FEATURES = NUM_COLS + CAT_COLS


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data types. Safe to run on training data, batch uploads, or one row."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data and add engineered features."""
    df = clean(df)
    df["num_addons"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    # tenure can be 0 for brand-new customers, so clip the divisor at 1
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    return df


def load_training_data(path):
    """Return (X, y) for training. Drops the ID and converts the target to 0/1."""
    raw = pd.read_csv(path)
    y = (raw["Churn"] == "Yes").astype(int)
    X = add_features(raw.drop(columns=["customerID", "Churn"]))
    return X, y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
