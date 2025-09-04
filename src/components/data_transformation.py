import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class Transformation:
    processor_path: str = os.path.join('Artifacts', 'Processor.pkl')

class DataTransformation:
    def __init__(self):
        self.preprocess_config = Transformation()

    def get_data_transformation_obj(self):
        """Create ColumnTransformer for numeric scaling."""
        try:
            num_features = ['sensor_reading', 'noise_level', 'rolling_avg',
                            'reading_diff', 'pga', 'snr']

            num_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            logging.info(f'Numeric features: {num_features}')

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_features)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path_cl, test_path_cl, train_path_rg, test_path_rg):
        try:
            
            preprocessor = self.get_data_transformation_obj()
            # Load datasats
            train_cl = pd.read_csv(train_path_cl)
            test_cl = pd.read_csv(test_path_cl)
            train_rg = pd.read_csv(train_path_rg)
            test_rg = pd.read_csv(test_path_rg)

            # Targets
            reg_target = 'ttf_seconds'
            cl_target = 'p_wave_detected'


            feature_cols = ['sensor_reading', 'noise_level', 'rolling_avg',
                            'reading_diff', 'pga', 'snr']


            # Classification transformation
            X_train_cl = train_cl[feature_cols]
            y_train_cl = train_cl[cl_target]
            X_test_cl = test_cl[feature_cols]
            y_test_cl = test_cl[cl_target]

            # Regression transformation
            X_train_rg = train_rg[feature_cols]
            y_train_rg = np.log1p(train_rg[reg_target])               
            X_test_rg = test_rg[feature_cols]
            y_test_rg = np.log1p(test_rg[reg_target])

            # Fit on classification train
            X_train_cl_t = preprocessor.fit_transform(X_train_cl)
            X_test_cl_t = preprocessor.transform(X_test_cl)

            # Transform regression
            X_train_rg_t = preprocessor.transform(X_train_rg)
            X_test_rg_t = preprocessor.transform(X_test_rg)

            # Combine features + targets into numpy arrays
            train_arr_cl = np.c_[X_train_cl_t, y_train_cl.to_numpy()]
            test_arr_cl = np.c_[X_test_cl_t, y_test_cl.to_numpy()]
            train_arr_rg = np.c_[X_train_rg_t, y_train_rg.to_numpy()]
            test_arr_rg = np.c_[X_test_rg_t, y_test_rg.to_numpy()]

            # Save the processor
            save_object(self.preprocess_config.processor_path, preprocessor)
            logging.info("Preprocessing object saved successfully.")

            return (train_arr_cl, 
                    test_arr_cl, 
                    train_arr_rg, 
                    test_arr_rg, 
                )

        except Exception as e:
            raise CustomException(e, sys)
