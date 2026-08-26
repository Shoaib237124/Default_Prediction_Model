import requests
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Credit Risk Assessment Engine",
    page_icon="🏦",
    layout="centered",
)

FASTAPI_URL = "http://127.0.0.1:8000/predict"

# ============================================================
# CUSTOM CSS FOR DECISION CARDS & UI
# ============================================================
st.markdown(
    """
<style>
    .header-box {
        border: 2px solid #334155;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        background-color: #0f172a;
        color: #f8fafc;
        margin-bottom: 25px;
    }
    .metric-card {
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        background-color: #1e293b;
        color: #ffffff;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin-top: 5px;
    }
    .banner-reject {
        background-color: #7f1d1d;
        color: #fecaca;
        border: 1px solid #ef4444;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .banner-approve {
        background-color: #14532d;
        color: #bbf7d0;
        border: 1px solid #22c55e;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .driver-card-red {
        border-left: 4px solid #ef4444;
        background-color: #1e293b;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 0 6px 6px 0;
    }
    .driver-card-green {
        border-left: 4px solid #22c55e;
        background-color: #1e293b;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 0 6px 6px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div class="header-box">
    <h2>🏦 CREDIT DECISION ENGINE</h2>
    <p style="margin:0; color: #94a3b8;">AI-Powered Credit Risk Assessment</p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# APPLICANT INPUT FORM
# ============================================================
with st.form("loan_application_form"):
    st.subheader("📋 Loan Application Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Financial Information**")
        amt_income = st.number_input(
            "Total Annual Income ($)", min_value=1000.0, value=120000.0, step=5000.0
        )
        amt_credit = st.number_input(
            "Requested Credit Amount ($)", min_value=1000.0, value=450000.0, step=5000.0
        )
        amt_annuity = st.number_input(
            "Annuity / Monthly Payment ($)", min_value=100.0, value=24000.0, step=500.0
        )
        amt_goods_price = st.number_input(
            "Goods Price ($)", min_value=0.0, value=400000.0, step=5000.0
        )

        st.markdown("**External Bureau Scores**")
        ext_source_1 = st.number_input(
            "External Source Score 1", 0.0, 1.0, 0.25, 0.01
        )
        ext_source_2 = st.number_input(
            "External Source Score 2", 0.0, 1.0, 0.30, 0.01
        )
        ext_source_3 = st.number_input(
            "External Source Score 3", 0.0, 1.0, 0.20, 0.01
        )

    with col2:
        st.markdown("**Demographics**")
        gender_display = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )
        code_gender = "F" if gender_display == "Female" else "M"

        age_years = st.number_input(
            "Age (Years)", min_value=18, max_value=100, value=38, step=1
        )
        years_employed = st.number_input(
            "Years Employed", min_value=0.0, max_value=60.0, value=3.5, step=0.5
        )
        name_family_status = st.selectbox(
            "Family Status",
            ["Married", "Single / not married", "Civil marriage", "Separated"],
        )
        name_education_type = st.selectbox(
            "Education Type",
            [
                "Higher education",
                "Secondary / secondary special",
                "Incomplete higher",
            ],
        )

        st.markdown("**Historical Financial Behavior**")
        late_payment_rate = st.number_input(
            "Late Payment Rate", 0.0, 1.0, 0.15, 0.01
        )
        bureau_debt_ratio = st.number_input(
            "Bureau Debt-to-Credit Ratio", 0.0, 5.0, 0.65, 0.05
        )
        prev_refusal_rate = st.number_input(
            "Previous Refusal Rate", 0.0, 1.0, 0.40, 0.05
        )
        prev_amt_down = st.number_input(
            "Prev. Down Payment Max", min_value=0.0, value=0.0
        )
        pos_cnt_instalment = st.number_input(
            "POS Instalments Future Mean", min_value=0.0, value=8.0
        )
        pos_months_bal = st.number_input(
            "POS Months Balance Max", max_value=0.0, value=-1.0
        )

    submit_btn = st.form_submit_button(
        "Evaluate Application", use_container_width=True
    )

# ============================================================
# EVALUATION & DECISION DISPLAY
# ============================================================
if submit_btn:
    # ----------------------------------------------------
    # BEHIND-THE-SCENES FEATURE TRANSFORMATIONS
    # ----------------------------------------------------
    calculated_credit_annuity_ratio = (
        amt_credit / amt_annuity if amt_annuity > 0 else 0.0
    )
    calculated_annuity_income_ratio = (
        amt_annuity / amt_income if amt_income > 0 else 0.0
    )

    # Convert positive user inputs into negative days expected by the model
    days_birth = -int(age_years * 365.25)
    days_employed = -float(years_employed * 365.25)

    payload = {
        "CODE_GENDER": code_gender,
        "EXT_SOURCE_1": ext_source_1,
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "AMT_GOODS_PRICE": amt_goods_price,
        "CREDIT_ANNUITY_RATIO": round(calculated_credit_annuity_ratio, 4),
        "ANNUITY_INCOME_RATIO": round(calculated_annuity_income_ratio, 4),
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "NAME_FAMILY_STATUS": name_family_status,
        "NAME_EDUCATION_TYPE": name_education_type,
        "LATE_PAYMENT_RATE": late_payment_rate,
        "BUREAU_DEBT_TO_CREDIT_RATIO_max": bureau_debt_ratio,
        "PREV_AMT_DOWN_PAYMENT_max": prev_amt_down,
        "PREV_REFUSAL_RATE": prev_refusal_rate,
        "POS_CNT_INSTALMENT_FUTURE_mean": pos_cnt_instalment,
        "POS_MONTHS_BALANCE_max": pos_months_bal,
    }

with st.spinner("Processing credit decision..."):
    try:
        response = requests.post(FASTAPI_URL, json=payload, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Parse backend schema directly
            prob_raw = data.get("default_probability", 0.0)
            default_risk_pct = f"{round(prob_raw * 100, 2)}%"
            risk_level = data.get("risk_level", "UNKNOWN").upper()
            decision = data.get("decision", "REJECT").upper()
            drivers = data.get("top_5_shap_drivers", [])

            st.markdown("### CREDIT RISK ASSESSMENT")

            # Summary Metric Cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div style="color: #94a3b8; font-size: 14px;">Default Risk</div>
                    <div class="metric-value">{default_risk_pct}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div style="color: #94a3b8; font-size: 14px;">Risk Level</div>
                    <div class="metric-value">{risk_level}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div style="color: #94a3b8; font-size: 14px;">Decision</div>
                    <div class="metric-value">{decision}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Progress Bar
            st.write("")
            st.write("**Estimated Default Risk**")
            st.progress(min(max(prob_raw, 0.0), 1.0))

            # Status Banner
            banner_class = (
                "banner-reject" if decision == "REJECT" else "banner-approve"
            )
            banner_icon = "⚠️" if decision == "REJECT" else "✅"
            st.markdown(
                f"""
            <div class="{banner_class}">
                {banner_icon} Application {decision.title()}ed — Risk Level: {risk_level}
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("### 🔍 WHY DID THE MODEL MAKE THIS DECISION?")

            # SHAP Drivers Mapping
            for item in drivers:
                shap_val = item.get("shap_value", 0.0)
                feature_raw = item.get("feature", "Unknown Feature")

                # Clean feature names (remove preprocessor prefixes)
                clean_name = (
                    feature_raw.replace("num__", "")
                    .replace("cat__", "")
                    .replace("_", " ")
                    .title()
                )

                is_risk_increase = shap_val > 0
                card_class = (
                    "driver-card-red"
                    if is_risk_increase
                    else "driver-card-green"
                )
                bullet_icon = "🔴" if is_risk_increase else "🟢"
                impact_text = (
                    "Increased default risk"
                    if is_risk_increase
                    else "Decreased default risk"
                )
                sign = "+" if is_risk_increase else ""

                st.markdown(
                    f"""
                <div class="{card_class}">
                    <div style="font-weight: bold; font-size: 16px;">{bullet_icon} {clean_name}</div>
                    <div style="color: #94a3b8; font-size: 13px;">{impact_text}</div>
                    <div style="font-weight: bold; font-size: 13px; margin-top: 3px;">SHAP: {sign}{round(shap_val, 4)}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        else:
            st.error(f"API Error ({response.status_code}): {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Connection Refused: Unable to reach FastAPI server at "
            f"{FASTAPI_URL}. Ensure uvicorn is running."
        )
    except Exception as e:
        st.error(f"UI Rendering Error: {str(e)}")