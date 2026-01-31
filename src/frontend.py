import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =============================
# Load trained model
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Artifacts" / "model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipeline = load_model()

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="Loan Risk Prediction",
    layout="centered",
    page_icon="💳"
)

st.title("💳 Loan Risk Prediction App")
st.caption("Predict loan applicant risk using a trained ML pipeline")

# =============================
# Helper functions
# =============================
def risk_badge(risk):
    colors = {
        "Low": "🟢 LOW RISK",
        "Medium": "🟠 MEDIUM RISK",
        "High": "🔴 HIGH RISK"
    }
    return colors.get(risk, risk)

REQUIRED_COLS = [
    "Age", "Income", "EmploymentType", "ResidenceType",
    "CreditScore", "LoanAmount", "LoanTerm", "PreviousDefault"
]

# =============================
# Tabs
# =============================
tab1, tab2, tab3 = st.tabs(
    ["🧍 Single Prediction", "📂 CSV Upload", "🌐 Google Sheets"]
)

# =====================================================
# 1️⃣ Single Entry Prediction
# =====================================================
with tab1:
    st.subheader("Single Applicant Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Annual Income", min_value=0, value=50000)
        employment = st.selectbox(
            "Employment Type",
            ["Salaried", "Self-employed", "Unemployed"]
        )
        residence = st.selectbox(
            "Residence Type",
            ["Owned", "Rented", "Parental Home"]
        )

    with col2:
        credit_score = st.number_input("Credit Score", 300, 900, 700)
        loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
        loan_term = st.number_input("Loan Term (months)", 12, 360, 60)
        previous_default = st.selectbox("Previous Default", ["Yes", "No"])

    if st.button("🔮 Predict Risk", use_container_width=True):
        input_df = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "EmploymentType": employment,
            "ResidenceType": residence,
            "CreditScore": credit_score,
            "LoanAmount": loan_amount,
            "LoanTerm": loan_term,
            "PreviousDefault": previous_default
        }])

        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]

        st.success(f"**Prediction:** {risk_badge(prediction)}")

        prob_df = pd.DataFrame({
            "Risk Category": pipeline.classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        colA, colB = st.columns(2)

        with colA:
            st.markdown("### 📊 Probability Table")
            st.dataframe(prob_df, use_container_width=True)

        with colB:
            st.markdown("### 📈 Probability Chart")
            st.bar_chart(prob_df.set_index("Risk Category"))

        # Download report
        report = input_df.copy()
        report["PredictedRisk"] = prediction
        csv = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction Report",
            csv,
            "single_prediction.csv",
            "text/csv"
        )

# =====================================================
# 2️⃣ CSV Upload Prediction
# =====================================================
with tab2:
    st.subheader("Batch Prediction (CSV Upload)")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.markdown("### 👀 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        if not all(col in df.columns for col in REQUIRED_COLS):
            st.error(
                f"CSV must contain columns:\n{', '.join(REQUIRED_COLS)}"
            )
        else:
            df["PredictedRisk"] = pipeline.predict(df)

            st.success("✅ Predictions completed")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions",
                csv,
                "batch_predictions.csv",
                "text/csv"
            )

# =====================================================
# 3️⃣ Google Sheets Integration
# =====================================================
with tab3:
    st.subheader("Google Sheets Integration")
    st.info("Uses Google Form responses stored in Google Sheets")

    sheet_name = st.text_input("Google Sheet Name")

    if st.button("📥 Fetch & Predict"):
        try:
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials

            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"
            ]

            creds = ServiceAccountCredentials.from_json_keyfile_dict(
                st.secrets["gcp_service_account"], scope
            )

            client = gspread.authorize(creds)
            sheet = client.open(sheet_name).sheet1

            data = sheet.get_all_records()
            df_google = pd.DataFrame(data)

            if not all(col in df_google.columns for col in REQUIRED_COLS):
                st.error("Google Sheet schema mismatch.")
            else:
                df_google["PredictedRisk"] = pipeline.predict(df_google)
                st.success("✅ Predictions completed")
                st.dataframe(df_google, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# =============================
# Footer
# =============================
with st.expander("ℹ️ Model Info"):
    st.write(
        """
        - Model loaded from `Artifacts/model.joblib`
        - Pipeline includes preprocessing + classifier
        - Supports single & batch predictions
        """
    )
