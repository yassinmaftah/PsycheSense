import json
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from src.preprocessing import CV_RESULTS_PATH, METRICS_PATH, MODEL_PATH
from src.predict import load_model, predict_one

st.set_page_config(page_title="PsycheSense", layout="wide")

st.title("PsycheSense")


@st.cache_resource
def get_model():
    return load_model()


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as file:
        return json.load(file)


tab_perf, tab_predict = st.tabs(["Model performance", "Test the model"])

with tab_perf:
    cv_results = read_json(CV_RESULTS_PATH)
    metrics = read_json(METRICS_PATH)

    if cv_results is None or metrics is None:
        st.error("Results not found. Run the training and evaluation first.")
    else:
        st.subheader("Model comparison (cross-validation on train set)")
        rows = []
        for name, info in cv_results["models"].items():
            rows.append({"Model": name, "CV F1 macro": round(info["cv_f1_macro"], 4)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.write("Selected model:", cv_results["best_model"])

        st.subheader("Metrics on the independent test set")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", round(metrics["accuracy"], 3))
        col2.metric("Precision (macro)", round(metrics["precision_macro"], 3))
        col3.metric("Recall (macro)", round(metrics["recall_macro"], 3))
        col4.metric("F1 (macro)", round(metrics["f1_macro"], 3))
        col5.metric("ROC-AUC", round(metrics["roc_auc_ovr"], 3))

        st.subheader("Per-class metrics")
        class_rows = []
        for label in metrics["labels"]:
            values = metrics["per_class"][label]
            class_rows.append({
                "Class": label,
                "Precision": round(values["precision"], 3),
                "Recall": round(values["recall"], 3),
                "F1": round(values["f1"], 3),
            })
        st.dataframe(pd.DataFrame(class_rows), hide_index=True)

        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=metrics["labels"], yticklabels=metrics["labels"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Real")
        st.pyplot(fig)

with tab_predict:
    st.subheader("Enter the student information")

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Run the training first.")
    else:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            age = st.number_input("Age", 16, 40, 21)
            gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
            year = st.number_input("Year of study", 1, 6, 2)
            sleep = st.number_input("Sleep hours", 0.0, 14.0, 7.0, step=0.1)
            study = st.number_input("Study hours per week", 0.0, 80.0, 20.0, step=0.5)

        with col_b:
            gpa = st.number_input("GPA (0 to 4)", 0.0, 4.0, 2.5, step=0.05)
            social = st.number_input("Social media hours", 0.0, 24.0, 6.0, step=0.1)
            screen = st.number_input("Screen time hours", 0.0, 24.0, 8.5, step=0.1)
            stress = st.number_input("Financial stress (0 to 10)", 0.0, 10.0, 6.0, step=0.1)
            support = st.number_input("Family support (0 to 10)", 0.0, 10.0, 6.5, step=0.1)

        with col_c:
            attendance = st.number_input("Attendance rate (%)", 0.0, 100.0, 85.0, step=0.5)
            activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"])
            diet = st.selectbox("Diet quality", ["Poor", "Average", "Good"])
            job = st.selectbox("Part-time job", ["No", "Yes"])
            history = st.selectbox("Previous mental health history", ["No", "Yes"])

        if st.button("Predict"):
            if social > screen:
                st.warning("Social media hours cannot exceed screen time. "
                           "Using screen time as the social media value.")
                social = screen

            student = {
                "Age": age,
                "Gender": gender,
                "Year_of_Study": year,
                "Sleep_Hours": sleep,
                "Study_Hours_Per_Week": study,
                "GPA": gpa,
                "Social_Media_Hours": social,
                "Screen_Time_Hours": screen,
                "Physical_Activity": activity,
                "Financial_Stress": stress,
                "Family_Support": support,
                "Attendance_Rate": attendance,
                "Part_Time_Job": job,
                "Diet_Quality": diet,
                "Previous_Mental_Health_History": history,
            }

            try:
                model = get_model()
                label, probabilities = predict_one(model, student)

                st.subheader("Estimated risk level: " + label)
                proba_table = pd.DataFrame({
                    "Class": list(probabilities.keys()),
                    "Probability": list(probabilities.values()),
                }).set_index("Class")
                st.bar_chart(proba_table)
            except Exception as error:
                st.error("Prediction failed: " + str(error))