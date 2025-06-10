# predict.py
import pandas as pd
import numpy as np
import joblib
import lime.lime_tabular
import matplotlib.pyplot as plt

# Load encoded training data, encoders, and model
X_encoded = joblib.load("X_encoded.pkl")  # encoded features
X_raw = joblib.load("X_raw.pkl")  # raw features with strings
encoders = joblib.load("encoders.pkl")
model = joblib.load("lgbm_model.pkl")

# Define input sample (raw format)
input_data = {
    "Transaction.Amount": 88.04,
    "Customer.Age": 29,
    "Account.Age.Days": 44,
    "Transaction.Hour": 14,
    "source": "SEO",
    "browser": "IE",
    "sex": "M",
    "Payment.Method": "debit card",
    "Product.Category": "toys and games",
    "Quantity": 3,
    "Device.Used": "tablet",
    "Address.Match": 0
}
input_df_raw = pd.DataFrame([input_data])
input_df = input_df_raw.copy()

# Encode input
for col in input_df.columns:
    if col in encoders:
        le = encoders[col]
        input_df[col] = input_df[col].apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
        )

# Prediction
pred = model.predict(input_df)[0]
result = "Fraudulent Transaction" if pred == 1 else "Safe Transaction"
print("Prediction:", result)

proba = model.predict_proba(input_df)[0]  # Get [non-fraud, fraud]
pred = model.predict(input_df)[0]         # 0 or 1 (prediction)

confidence = round(proba[pred] * 100, 2)  # Confidence in the predicted class
risk_score = round(proba[1] * 100, 2)     # Risk of fraud (class 1)

print(f"Confidence: {confidence}%")
print(f"Risk Score: {risk_score}%")

# Categorical feature indices (from raw data)
categorical_features = [
    i for i, col in enumerate(X_raw.columns)
    if X_raw[col].dtype == "object"
]

# LIME Explainer on encoded training data
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_encoded.values,
    feature_names=X_encoded.columns.tolist(),
    class_names=["Not Fraudulent", "Fraudulent"],
    mode="classification",
    categorical_features=categorical_features,
    discretize_continuous=True
)

# LIME Prediction Function
def predict_fn_lime(data):
    temp_df = pd.DataFrame(data, columns=X_encoded.columns)
    return model.predict_proba(temp_df)

# Explanation
exp = explainer.explain_instance(
    data_row=input_df.values[0],
    predict_fn=predict_fn_lime,
    num_features=10
)

# Save as PNG
fig = exp.as_pyplot_figure()
fig.tight_layout()
fig.savefig("lime_explanation.png")
print("LIME explanation saved as lime_explanation.png")
