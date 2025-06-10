# fraud_detection.py

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("ecomdataset.csv")
df = df.drop(columns=["Transaction.Date"])  # Drop unneeded columns

# Separate features and target
X = df.drop("Is.Fraudulent", axis=1)
y = df["Is.Fraudulent"]

# Save raw features before encoding
X_raw = X.copy()
joblib.dump(X_raw, "X_raw.pkl")  # for LIME

# Encode categorical features
cat_cols = X.select_dtypes(include=["object"]).columns
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Save encoded data and encoders
joblib.dump(X, "X_encoded.pkl")
joblib.dump(encoders, "encoders.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Train LightGBM model
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "lgbm_model.pkl")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get predicted probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LightGBM Fraud Detection')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('lgbm_roc_curve.png')  # Save as PNG
plt.close()
