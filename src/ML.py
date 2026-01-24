import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("C:\\Users\\AIML\\Desktop\\DSCE_MLOPs\\Data\\loan_risk_data.csv")  # change filename if needed

# -----------------------------
# 2. Separate features & target
# -----------------------------
X = df.drop("RiskCategory", axis=1)
y = df["RiskCategory"]

# -----------------------------
# 3. Identify column types
# -----------------------------
numerical_features = [
    "Age", "Income", "CreditScore",
    "LoanAmount", "LoanTerm"
]

categorical_features = [
    "EmploymentType", "ResidenceType", "PreviousDefault"
]#Age,Income,EmploymentType,ResidenceType,CreditScore,LoanAmount,LoanTerm,PreviousDefault,RiskCategory

# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# -----------------------------
# 5. Multiclass Logistic Regression
# -----------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000
    ))
])

# -----------------------------
# 6. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 7. Train model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 8. Evaluate model
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
import joblib

joblib.dump(model, "model.joblib")
