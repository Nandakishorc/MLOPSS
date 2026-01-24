import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Load trained model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Artifacts" / "model.joblib"
pipeline = joblib.load(MODEL_PATH)


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Loan Risk Prediction",
    layout="centered"
)

st.title("Loan Risk Prediction App")
st.write("Predict whether a loan applicant is Low, Medium, or High Risk.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Single Entry", "CSV Upload", "Google Form (Optional)"])

# =====================================================
# 1️⃣ Single Entry Prediction
# =====================================================
with tab1:
    st.subheader("Single Entry Prediction")

    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=0, value=50000)

    employment = st.selectbox(
        "Employment Type",
        ["Salaried", "Self-employed", "Unemployed"]
    )

    residence = st.selectbox(
        "Residence Type",
        ["Owned", "Rented", "Parental Home"]
    )

    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
    loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=60)

    previous_default = st.selectbox(
        "Previous Default",
        ["Yes", "No"]
    )

    if st.button("Predict Risk"):
        input_df = pd.DataFrame({
            "Age": [age],
            "Income": [income],
            "EmploymentType": [employment],
            "ResidenceType": [residence],
            "CreditScore": [credit_score],
            "LoanAmount": [loan_amount],
            "LoanTerm": [loan_term],
            "PreviousDefault": [previous_default]
        })

        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]

        st.success(f"Predicted Risk Category: **{prediction}**")

        prob_df = pd.DataFrame({
            "Risk Category": pipeline.classes_,
            "Probability": probabilities
        })

        st.bar_chart(prob_df.set_index("Risk Category"))

# =====================================================
# 2️⃣ CSV Upload Prediction
# =====================================================
with tab2:
    st.subheader("Batch Prediction via CSV Upload")

    uploaded_file = st.file_uploader(
        "Upload a CSV file with customer data",
        type=["csv"]
    )

    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df_uploaded.head())

        required_cols = [
            "Age", "Income", "EmploymentType", "ResidenceType",
            "CreditScore", "LoanAmount", "LoanTerm", "PreviousDefault"
        ]

        if not all(col in df_uploaded.columns for col in required_cols):
            st.error("CSV file does not match required schema.")
        else:
            df_uploaded["PredictedRisk"] = pipeline.predict(df_uploaded)

            st.success("Predictions complete!")
            st.dataframe(df_uploaded)

            csv = df_uploaded.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

# =====================================================
# 3️⃣ Google Form / Google Sheets Integration (Optional)
# =====================================================
with tab3:
    st.subheader("Google Form / Google Sheets Integration (Optional)")
    st.write("Fetch Google Form responses stored in Google Sheets and predict risk.")

    st.info("Requires `gspread` and Google Service Account credentials.")

    if st.button("Fetch & Predict Google Form Data"):
        try:
            import gspread
            from oauth2client.service_account import ServiceAccountCredentials

            sheet_name = st.text_input("Enter Google Sheet Name")

            if sheet_name:
                creds = ServiceAccountCredentials.from_json_keyfile_dict(
                    st.secrets["gcp_service_account"],
                    [
                        "https://spreadsheets.google.com/feeds",
                        "https://www.googleapis.com/auth/drive"
                    ]
                )

                client = gspread.authorize(creds)
                sheet = client.open(sheet_name).sheet1
                data = sheet.get_all_records()

                df_google = pd.DataFrame(data)
                df_google["PredictedRisk"] = pipeline.predict(df_google)

                st.success("Predictions complete!")
                st.dataframe(df_google)

        except Exception as e:
            st.error(f"Error: {e}")
