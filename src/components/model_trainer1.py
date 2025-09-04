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

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models_classification


@dataclass
class ModelTrainerConfig:
    train_model_file_path: str = os.path.join('Artifacts', 'p_wave_detection_model.pkl')


class ClassificationTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    # model_trainer1.py (replace initiate_model_training method)
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays into features and targets for classification")

            # train_array/test_array are numpy arrays: features + last column = target (p_wave_detected)
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1].astype(int)

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1].astype(int)

            models = {
                'Logistic Regression': LogisticRegression(max_iter=500),
                'Random Forest': RandomForestClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Support Vector Classifier': SVC(),
                'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                'CatBoost': CatBoostClassifier(verbose=0),
                'AdaBoost': AdaBoostClassifier()
            }

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
            model_report = evaluate_models_classification(
                X_train=X_train, Y_train=y_train,
                X_test=X_test, Y_test=y_test,
                models=models, param=params
            )

            best_model_name = max(model_report, key=model_report.get)
            best_score = model_report[best_model_name]
            logging.info(f"Best classification model according to CV/val: {best_model_name} -> {best_score}")

            # Fit chosen model on training data
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            # Save trained classifier
            save_object(self.config.train_model_file_path, best_model)
            logging.info(f"Saved best classification model: {self.config.train_model_file_path}")

            preds = best_model.predict(X_test)
            final_acc = accuracy_score(y_test, preds)
            logging.info(f"Final classification accuracy on test: {final_acc}")

            return final_acc

        except Exception as e:
            raise CustomException(e, sys)
