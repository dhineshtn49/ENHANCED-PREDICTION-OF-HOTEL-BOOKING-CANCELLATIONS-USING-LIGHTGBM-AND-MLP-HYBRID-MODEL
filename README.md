import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

print("Loading dataset...")
filename = "hotel_bookings.csv"
df = pd.read_csv(filename)

print("Preprocessing data...")
df.drop(['reservation_status', 'reservation_status_date'], axis=1, inplace=True)
df.fillna(0, inplace=True)

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

X = df.drop('is_canceled', axis=1)
y = df['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

print("Training MLP...")
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
mlp_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)
y_pred_mlp = mlp_model.predict(X_test)

acc_lgb = accuracy_score(y_test, y_pred_lgb)
acc_mlp = accuracy_score(y_test, y_pred_mlp)

print(f"\nLightGBM Accuracy: {acc_lgb}")
print(f"MLP Accuracy: {acc_mlp}")
print("\nClassification Report (LightGBM):")
print(classification_report(y_test, y_pred_lgb))

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# 2. Feature Importance
lgb.plot_importance(lgb_model, max_num_features=10, importance_type='gain')
plt.title("Top 10 Feature Importance (LightGBM)")
plt.show()

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lgb)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Canceled", "Canceled"], yticklabels=["Not Canceled", "Canceled"])
plt.title("Confusion Matrix (LightGBM)")
plt.show()

# 4. SHAP Summary & Bar Plot
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list) and len(shap_values) == 2:
    shap.summary_plot(shap_values[1], X_test)
    shap.summary_plot(shap_values[1], X_test, plot_type="bar")
else:
    shap.summary_plot(shap_values, X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")

# 5. Accuracy Comparison Plot
models = ["LightGBM", "MLP"]
accuracies = [acc_lgb, acc_mlp]

plt.bar(models, accuracies, color=['green', 'orange'])
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.0)
plt.title("Model Accuracy Comparison")
plt.show()
