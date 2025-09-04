from src.exception import CustomException
from src.logger import logging
import dill
import os
import numpy as np
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score


def save_object(path, obj):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        dill.dump(obj, file=file)


def evaluate_models_regresion(X_train, Y_train, X_test, Y_test, models, param):
    try:
        report = {}
        for name, model in models.items():
            gs = GridSearchCV(model, param[name], cv=3)
            gs.fit(X_train, Y_train)
            model.set_params(**gs.best_params_)

            model.fit(X_train, Y_train)
            y_test_pred =np.expm1( model.predict(X_test))
            test_model_score = r2_score(Y_test, y_test_pred)
            report[name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models_classification(X_train, Y_train, X_test, Y_test, models, param):
    try:
        report = {}
        for name, model in models.items():
            gs = GridSearchCV(model, param[name], cv=3)
            gs.fit(X_train, Y_train)
            model.set_params(**gs.best_params_)

            model.fit(X_train, Y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = accuracy_score(Y_test, y_test_pred)
            report[name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)
