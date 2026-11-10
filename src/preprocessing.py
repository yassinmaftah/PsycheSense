import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_mental_health.csv")

TARGET = "Mental_Health_Risk"
TEST_SIZE = 0.2
RANDOM_STATE = 42
numeric_columns = ["Age","Year_of_Study","Sleep_Hours","Study_Hours_Per_Week","GPA","Social_Media_Hours","Screen_Time_Hours"
                   ,"Financial_Stress","Family_Support","Attendance_Rate"]
ordinal_columns = ["Physical_Activity","Diet_Quality"]
binary_columns = ["Part_Time_Job","Previous_Mental_Health_History"]
nominal_columns = ["Gender"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns='Student_ID')
    return df


def split_data(df):
    y = df[TARGET]
    X = df.drop(columns=TARGET)

    return train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)


def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('encoder',OrdinalEncoder(
                                        categories=[["Low", "Moderate", "High"],
                                                    ["Poor", "Average", "Good"]],
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1
                                        ))
    ])

    binary_pipeline = Pipeline(
        steps=[
            ('encoder',OneHotEncoder(drop="if_binary", handle_unknown="ignore",sparse_output=False))
        ]
    )

    nominal_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("ordinal", ordinal_pipeline, ordinal_columns),
        ("binary", binary_pipeline, binary_columns),
        ("nominal", nominal_pipeline, nominal_columns)
    ])
    
    return preprocessor