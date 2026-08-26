import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

app = FastAPI(
    title="Credit Risk Inference Engine - Top 20 Features", version="1.0.0"
)

# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================
model = XGBClassifier()
model.load_model("credit_risk_xgboost.json")

preprocessor = joblib.load("credit_risk_preprocessor.joblib")
explainer = shap.TreeExplainer(model.get_booster())

# Extract exact input columns expected by the preprocessor during fit
EXPECTED_FEATURES = list(
    getattr(
        preprocessor,
        "feature_names_in_",
        [
            col
            for trans in preprocessor.transformers_
            if trans[2] != "drop"
            for col in (
                trans[2]
                if isinstance(trans[2], list)
                else preprocessor.feature_names_in_
            )
        ],
    )
)

OPTIMAL_THRESHOLD = 0.5


# ============================================================
# REQUEST SCHEMA
# ============================================================
class LoanApplicationRequest(BaseModel):
    EXT_SOURCE_1: float = Field(..., ge=0.0, le=1.0, example=0.55)
    EXT_SOURCE_2: float = Field(..., ge=0.0, le=1.0, example=0.62)
    EXT_SOURCE_3: float = Field(..., ge=0.0, le=1.0, example=0.48)

    AMT_CREDIT: float = Field(..., example=450000.0)
    AMT_ANNUITY: float = Field(..., example=24000.0)
    AMT_GOODS_PRICE: float = Field(..., example=400000.0)
    CREDIT_ANNUITY_RATIO: float = Field(..., example=18.75)
    ANNUITY_INCOME_RATIO: float = Field(..., example=0.20)

    DAYS_BIRTH: int = Field(..., example=-14000)
    DAYS_EMPLOYED: float = Field(..., example=-2500.0)
    NAME_FAMILY_STATUS: str = Field(..., example="Married")
    NAME_EDUCATION_TYPE: str = Field(..., example="Higher education")
    CODE_GENDER: str = Field(..., example="F")

    LATE_PAYMENT_RATE: float = Field(default=0.0, example=0.02)
    BUREAU_DEBT_TO_CREDIT_RATIO_max: float = Field(default=0.0, example=0.30)
    PREV_AMT_DOWN_PAYMENT_max: float = Field(default=0.0, example=0.0)
    PREV_REFUSAL_RATE: float = Field(default=0.0, example=0.0)
    POS_CNT_INSTALMENT_FUTURE_mean: float = Field(default=0.0, example=12.0)
    POS_MONTHS_BALANCE_max: float = Field(default=0.0, example=-1.0)


@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost", "shap": "enabled"}


# ============================================================
# PREDICTION
# ============================================================
@app.post("/predict")
def predict_credit_risk(payload: LoanApplicationRequest):
    try:
        # 1. Convert payload dict to DataFrame
        payload_dict = payload.model_dump()
        input_data = pd.DataFrame([payload_dict])

        # 2. Reindex against full training feature schema, filling missing columns with NaN
        full_df = input_data.reindex(columns=EXPECTED_FEATURES, fill_value=np.nan)

        # 3. Preprocess full dataset schema
        X_transformed = preprocessor.transform(full_df)

        # 4. Predict probabilities
        prob = float(model.predict_proba(X_transformed)[0, 1])
        prediction = int(prob >= OPTIMAL_THRESHOLD)

        if prob < 0.30:
            risk_level = "Low"
        elif prob < 0.60:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # 5. Compute SHAP values dynamically
        shap_values = explainer(X_transformed)
        values = (
            shap_values.values[0]
            if len(shap_values.values.shape) == 2
            else shap_values.values[0][:, 1]
        )

        feature_names = (
            preprocessor.get_feature_names_out()
            if hasattr(preprocessor, "get_feature_names_out")
            else EXPECTED_FEATURES
        )

        shap_impacts = [
            {"feature": feature, "shap_value": round(float(value), 4)}
            for feature, value in sorted(
                zip(feature_names, values),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
        ]

        return {
            "default_probability": round(prob, 4),
            "risk_level": risk_level,
            "decision": "REJECT" if prediction == 1 else "APPROVE",
            "threshold_used": OPTIMAL_THRESHOLD,
            "top_5_shap_drivers": shap_impacts,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))