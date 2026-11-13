import json
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV

from src.preprocessing import (
    load_data,split_data,build_preprocessor,
    MODELS_DIR,MODEL_PATH,CV_RESULTS_PATH,RANDOM_STATE,
)

def get_serch_settings():
    settings = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000),
            {"model__C":[0.01, 0.1, 1, 10, 100]}
        ),
        "Random Forest":(
                RandomForestClassifier(random_state=RANDOM_STATE),
                {
                    "model__max_depth":[3,5,8,None],
                    "model__min_samples_leaf":[1, 5, 10, 20],
                    "model__max_features": ["sqrt", 0.5],
                },
        ),"Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__n_estimators": [100, 200, 300],
            },
        ),
    }
    return settings

def train_models(X_train,y_train):
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)
    settings = get_serch_settings()
    searches = {}
    for name,(model,grid) in settings.items():
        print(f"start training with {name}")
        pip = Pipeline(
            steps=[
                ("preprocessor",build_preprocessor()),
                ("model",model),
            ]
        )
        grid_s = GridSearchCV(pip,grid,cv=cv,scoring="f1_macro",n_jobs=-1)
        grid_s.fit(X_train,y_train)
        
        print("best params:", grid_s.best_params_)
        print("best scors:", grid_s.best_score_)
        searches[name] = grid_s
        
    return searches


def choose_best(searches):
    best_name = None
    best_score = -1
    for name, search in searches.items():
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_name = name
    return best_name


def save_results(grid_s,best_name):
    os.makedirs(MODELS_DIR,exist_ok=True)
    joblib.dump(grid_s[best_name].best_estimator_, MODEL_PATH)
    
    results = {"best_model" : best_name, "models":{}}
    for name in grid_s:
        search = grid_s[name] 
        results['models'][name] = {
            "cv_f1_macro":float(search.best_score_),
            "best_param":search.best_params_,
        }
    
    with open(CV_RESULTS_PATH,"w") as file:
        json.dump(results,file,indent=4)
        
        
def main():
    df=load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print("Train shape: " , X_train.shape)
    
    grid_s = train_models(X_train,y_train)
    
    best_name = choose_best(grid_s)
    print("Best model:", best_name)
    
    save_results(grid_s, best_name)
    print("Saved:", MODEL_PATH)
    
if __name__ == "__main__":
    main()