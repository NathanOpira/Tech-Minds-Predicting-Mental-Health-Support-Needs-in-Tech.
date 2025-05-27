"""
Tech Minds: Predicting Mental Health Support Needs in the Tech Industry
Main Application Entry Point
"""

# Importing necessary modules
from src.data_loader import load_data
from src.features import preprocess_features
from src.train_model import train_all_models, evaluate_models
from src.explain import generate_shap_summary_plot

# Loading the data
print("🔄 Loading cleaned dataset...")
df = load_data('../data/processed/mental_health_cleaned.csv')

# Preparing features and target
print("⚙️ Preprocessing features and splitting data...")
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocess_features(df)

# Training models and returning trained ones
print("🧠 Training models...")
log_reg, rf_clf, xgb_clf = train_all_models(X_train, X_test, y_train)

# Evaluating model performances
print("\n📊 Evaluating Logistic Regression:")
evaluate_models(log_reg, X_test, y_test, X_test_scaled, scaled=True)

print("\n📊 Evaluating Random Forest:")
evaluate_models(rf_clf, X_test, y_test, X_test_scaled, scaled=False)

print("\n📊 Evaluating XGBoost:")
evaluate_models(xgb_clf, X_test, y_test, X_test_scaled, scaled=False)

# Generating SHAP summary plot for XGBoost
print("\n📈 Generating SHAP summary plot for XGBoost...")
generate_shap_summary_plot(xgb_clf, X_test, '../outputs/figures/shap_summary_xgb.png')

print("\n✅ All tasks completed successfully.")
