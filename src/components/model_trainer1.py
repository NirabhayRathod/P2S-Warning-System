import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from exception import CustomException
from logger import logging
from utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    train_model_file_path: str = os.path.join('artifacts', 'p_wave_detection_model.pkl')


class ClassificationTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting train and test arrays into features and targets for classification")

            # Features = all columns except last two
            X_train = train_array[:, :-2]
            y_train = train_array[:, -2]  # Classification target
            X_test = test_array[:, :-2]
            y_test = test_array[:, -2]

            # Classification models
            models = {
                'Logistic Regression': LogisticRegression(max_iter=500),
                'Random Forest': RandomForestClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Support Vector Classifier': SVC(),
                'XGBoost': XGBClassifier(),
                'CatBoost': CatBoostClassifier(verbose=0),
                'AdaBoost': AdaBoostClassifier()
            }

            # Hyperparameter grids
            params = {
                'Logistic Regression': {
                    'C': [0.1, 1, 10],
                    'solver': ['lbfgs', 'liblinear']
                },
                'Random Forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 5, 10]
                },
                'Decision Tree': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10]
                },
                'Gradient Boosting': {
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [50, 100, 200]
                },
                'K-Nearest Neighbors': {
                    'n_neighbors': [3, 5, 7]
                },
                'Support Vector Classifier': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                },
                'XGBoost': {
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [50, 100, 200]
                },
                'CatBoost': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                'AdaBoost': {
                    'learning_rate': [0.1, 0.5, 1.0],
                    'n_estimators': [50, 100, 200]
                }
            }

            logging.info("Evaluating classification models with provided hyperparameters")
            model_report = evaluate_models(
                X_train=X_train, Y_train=y_train,
                X_test=X_test, Y_test=y_test,
                models=models, param=params
            )

            # Select best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score <= 0.6:
                raise CustomException("No sufficiently good classification model found")

            logging.info(f"Best classification model selected: {best_model_name} with Accuracy: {best_model_score}")
            save_object(self.config.train_model_file_path, best_model)

            predictions = best_model.predict(X_test)
            final_score = accuracy_score(y_test, predictions)

            return final_score

        except Exception as e:
            raise CustomException(e, sys)
