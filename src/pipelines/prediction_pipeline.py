import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict_rg(self, features):
        """Predict Time-to-Failure (Regression)"""
        try:
            model_path = r'artifacts/ttf_model.pkl'
            preprocessor_path = r'artifacts/processor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            scaled_data = preprocessor.transform(features)
            return np.expm1(model.predict(scaled_data))

        except Exception as e:
            raise CustomException(e, sys)

    def predict_cl(self, features):
        """Predict P-wave detection (Classification)"""
        try:
            model_path = r'artifacts/p_wave_detection_model.pkl'
            preprocessor_path = r'artifacts/processor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            scaled_data = preprocessor.transform(features)
            return model.predict(scaled_data)

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 sensor_reading: float,
                 noise_level: float,
                 rolling_avg: float,
                 reading_diff: float,
                 pga: float,
                 snr: float):
        self.sensor_reading = sensor_reading
        self.noise_level = noise_level
        self.rolling_avg = rolling_avg
        self.reading_diff = reading_diff
        self.pga = pga
        self.snr = snr

    def get_data_as_dataframe(self):
        """Return input data as DataFrame"""
        try:
            data = {
                'sensor_reading': [self.sensor_reading],
                'noise_level': [self.noise_level],
                'rolling_avg': [self.rolling_avg],
                'reading_diff': [self.reading_diff],
                'pga': [self.pga],
                'snr': [self.snr],
            }
            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
