import os
import joblib 
import pandas as pd

from src.preprocessing import MODEL_PATH, feature_columns

LABELS = ["Low","Medium","High"]

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("we don't find file the model")
    return joblib.load(MODEL_PATH)


def predict_one(model, student):
    
    row = pd.DataFrame([student],columns=feature_columns)
    label = model.predict(row)[0]
    
    probabilities = model.predict_proba(row)[0]
    
    proba_by_class = dict(zip(model.classes_, probabilities))
    
    ordered = {}
    for name in LABELS:
        ordered[name] = float(proba_by_class[name])
    return label, ordered
    