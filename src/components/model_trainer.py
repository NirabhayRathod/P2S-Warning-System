import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from lightgbm import LGBMRegressor
from src.utils import save_object, evaluate_models_regresion

@dataclass
class ModelTrainerConfig:
    train_model_file_path: str = os.path.join('Artifacts', 'ttf_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays into features and targets (regression)")

            # train_array/test_array are numpy arrays: features + last column = target (ttf_seconds)
            X_train = train_array[:, :-1]
            Y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            Y_test = test_array[:, -1]

            models = {
               'Random Forest': RandomForestRegressor(),
               'Decision Tree': DecisionTreeRegressor(),
               'Gradient Boosting': GradientBoostingRegressor(),
               'Linear Regression': LinearRegression(),
               'K-Nearest Neighbors': KNeighborsRegressor(),
               'XGBoost': XGBRegressor(),
               'CatBoost': CatBoostRegressor(verbose=0),
               'AdaBoost': AdaBoostRegressor(),
            }

            params = {
                'Decision Tree': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                'Random Forest': {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'Gradient Boosting': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'Linear Regression': {},
                'K-Nearest Neighbors': {},
                'XGBoost': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'CatBoost': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                'AdaBoost': {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                
            }

            logging.info("Evaluating regression models with provided hyperparameters")
            model_report = evaluate_models_regresion(
                X_train=X_train, Y_train=Y_train,
                X_test=X_test, Y_test=Y_test,
                models=models, param=params
            )

            # model_report is assumed to be dict {model_name: score}
            best_model_name = max(model_report, key=model_report.get)
            best_score = model_report[best_model_name]
            logging.info(f"Best model according to CV/validation: {best_model_name} -> score: {best_score}")

            # Fit the selected model on full training set (to save & evaluate)
            best_model = models[best_model_name]
            best_model.fit(X_train, Y_train)

            # Save the trained model
            save_object(self.config.train_model_file_path, best_model)
            logging.info(f"Saved best regression model: {self.config.train_model_file_path}")

            # Final evaluation on test set
            predictions = best_model.predict(X_test)
            final_score = r2_score(Y_test, predictions)
            logging.info(f"Final R2 on test: {final_score}")

            return final_score

        except Exception as e:
            raise CustomException(e, sys)

