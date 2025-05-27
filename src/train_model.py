# Import necessary libraries.
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models(X_train_scaled, X_train, y_train):
    """
    Training Logistic Regression, Random Forest, and XGBoost models.
    """
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    log_reg.fit(X_train_scaled, y_train)
    rf_clf.fit(X_train, y_train)
    xgb_clf.fit(X_train, y_train)

    return log_reg, rf_clf, xgb_clf

def save_models(log_reg, rf_clf, xgb_clf, path_prefix="../models"):
    """
    Saving trained models using pickle.
    """
    with open(f"{path_prefix}/logistic_regression_model.pkl", 'wb') as f:
        pickle.dump(log_reg, f)
    with open(f"{path_prefix}/random_forest_model.pkl", 'wb') as f:
        pickle.dump(rf_clf, f)
    with open(f"{path_prefix}/xgboost_model.pkl", 'wb') as f:
        pickle.dump(xgb_clf, f)
