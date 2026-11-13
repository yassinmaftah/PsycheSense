import json
import os
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,confusion_matrix,
    classification_report,
    roc_auc_score,
)

from src.preprocessing import load_data,split_data,MODEL_PATH, METRICS_PATH

LABELS = ["Low", "Medium", "High"]

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("We don't find the model")
    return joblib.load(MODEL_PATH)


def evaluate_model(model,X_test,y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_proba,
                        multi_class="ovr",
                        labels=list(model.classes_))
    report = classification_report(y_test,y_pred,labels=LABELS,output_dict=True)
    matrix = confusion_matrix(y_test,y_pred,labels=LABELS)
    per_class = {}
    for label in LABELS :
        per_class[label] = {
            "precision":float(report[label]['precision']),
            "recall":float(report[label]['recall']),
            "f1":float(report[label]['f1-score']),
        }
        
        
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "roc_auc_ovr": float(auc),
        "labels": LABELS,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }
    return metrics

def save_metrics(metrics):
    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)
        
        
def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = load_model()
    metrics = evaluate_model(model,X_test,y_test)
    
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("F1 macro:", round(metrics["f1_macro"], 4))
    print("ROC-AUC (ovr):", round(metrics["roc_auc_ovr"], 4))
    print("Confusion matrix:", metrics["confusion_matrix"])
    
    save_metrics(metrics)
    print("Saved:", METRICS_PATH)
    
if __name__ == "__main__":
    main()