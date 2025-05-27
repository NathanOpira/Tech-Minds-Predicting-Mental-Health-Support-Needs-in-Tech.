# Importing required libraries.
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a classification model and prints metrics.

    Parameters:
        model: Trained classification model
        X_test: Features of the test set
        y_test: True labels of the test set
    """
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print(f"ROC AUC Score: {roc_auc_score(y_test, probas):.4f}")