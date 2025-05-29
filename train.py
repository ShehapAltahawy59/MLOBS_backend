import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import compute_class_weight
from sklearn.utils import compute_sample_weight
from sklearn.neighbors import KNeighborsClassifier
import joblib
from preprocessing import preprocess
import pickle
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

def train(features_train, labels_train,model):
    model.fit(features_train, labels_train)
    
    return model

if __name__ == "__main__":
    df = pd.read_csv("DataSet/hand_landmarks_data.csv")
    X_train, X_test, y_train, y_test = preprocess(df)
    mlflow.set_tracking_uri("http://localhost:5000")

    # Set the experiment name
    mlflow.set_experiment("hand_landmarks_predictions")

    # Logistic Regression
    with mlflow.start_run(run_name="Logistic Regression"):
        params = {
            "max_iter": 1000,
            "penalty": "l2",
            "C": 1.0,
            "solver": 'lbfgs',
            "multi_class": 'multinomial',
            "class_weight": 'balanced'
        }
        mlflow.log_params(params)

        log_reg = LogisticRegression(**params, random_state=42)
        model = train(X_train, y_train, log_reg)

        y_pred = model.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))

        conf_mat = confusion_matrix(y_test, y_pred)
        conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        conf_mat_disp.plot()
        plt.savefig("confusion_matrix_LR.png")
        mlflow.log_artifact("confusion_matrix_LR.png")

        joblib.dump(model, "./Models/LogisticRegression.pkl")
        mlflow.log_artifact("./Models/LogisticRegression.pkl")

    # SVM with GridSearchCV
    with mlflow.start_run(run_name="SVM with GridSearchCV"):
        svm_params = {
            "kernel": ["rbf"],
            "C": [0.01, 0.1, 1, 10, 100],
            "gamma": [0.01, 0.1, 1, 10, 100]
        }
        svm = SVC(class_weight="balanced")
        svm_gs = GridSearchCV(estimator=svm, param_grid=svm_params, cv=5)
        svm_gs.fit(X_train, y_train)

        best_model = svm_gs.best_estimator_
        y_pred = best_model.predict(X_test)

        mlflow.log_params(svm_gs.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))

        conf_mat = confusion_matrix(y_test, y_pred)
        conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        conf_mat_disp.plot()
        plt.savefig("confusion_matrix_SVM.png")
        mlflow.log_artifact("confusion_matrix_SVM.png")

        joblib.dump(best_model, "./Models/SVM_best_model.pkl")
        mlflow.log_artifact("./Models/SVM_best_model.pkl")

    # Random Forest with GridSearchCV
    with mlflow.start_run(run_name="Random Forest with GridSearchCV"):
        rf_params = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        }
        rf = RandomForestClassifier()
        rf_gs = GridSearchCV(estimator=rf, param_grid=rf_params, cv=5)
        rf_gs.fit(X_train, y_train)

        best_model = rf_gs.best_estimator_
        y_pred = best_model.predict(X_test)

        mlflow.log_params(rf_gs.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))

        conf_mat = confusion_matrix(y_test, y_pred)
        conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        conf_mat_disp.plot()
        plt.savefig("confusion_matrix_RF.png")
        mlflow.log_artifact("confusion_matrix_RF.png")

        joblib.dump(best_model, "./Models/RandomForest_best_model.pkl")
        mlflow.log_artifact("./Models/RandomForest_best_model.pkl")

    # KNN with GridSearchCV
    with mlflow.start_run(run_name="KNN with GridSearchCV"):
        knn_params = {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
            "p": [1, 2]
        }
        knn = KNeighborsClassifier()
        knn_gs = GridSearchCV(estimator=knn, param_grid=knn_params, cv=5)
        knn_gs.fit(X_train, y_train)

        best_model = knn_gs.best_estimator_
        y_pred = best_model.predict(X_test)

        mlflow.log_params(knn_gs.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))

        conf_mat = confusion_matrix(y_test, y_pred)
        conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        conf_mat_disp.plot()
        plt.savefig("confusion_matrix_KNN.png")
        mlflow.log_artifact("confusion_matrix_KNN.png")

        joblib.dump(best_model, "./Models/KNN_best_model.pkl")
        mlflow.log_artifact("./Models/KNN_best_model.pkl")
