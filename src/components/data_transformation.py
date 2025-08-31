import os 
import sys 
from exception import CustomException
from logger import logging
import pandas as pd
import numpy as np
from src.components import data_ingestion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from utils import save_object

@dataclass
class Transformation:
    processor_path : str=os.path.join('Artifacts','Processor.pkl')
    
class DataTransformation:
    def __init__(self):
       processor_obj=Transformation()
       
    def get_data_transformation_obj(self):
        try:          
            num_features = ['sensor_reading', 'noise_level', 'rolling_avg', 'reading_diff', 'pga', 'snr']
            num_pipeline = Pipeline(steps=[
               ('scaler', StandardScaler())
            ])
            logging.info('Numeric features are -> {num_features}')
        
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline , num_features)
            ])
            return preprocessor
            
        except Exception as e:
            raise CustomException(e ,sys)
        
    def initailize_data_transformation(self , train_path , test_path):
        try:
            train=pd.read_csv(train_path)
            test=pd.read_csv(test_path)
            
            preprocessor=self.get_data_transformation_obj()
            
            reg_target='ttf_seconds'
            cl_target='p_wave_detected'
            
            X_train=train.drop(columns=[reg_target , cl_target])
            y_train=train(columns=[reg_target ,cl_target])
            
            X_test=test.drop(columns=[reg_target , cl_target])
            y_test=test(columns=[reg_target ,cl_target])
            
            X_train_t=preprocessor.fit_transform(X_train)
            X_test_t=preprocessor.transform(X_test)
            
            
            train_arr = np.c_[X_train_t, y_train.to_numpy()]
            test_arr = np.c_[X_test_t, y_test.to_numpy()]

            save_object(file_path=self.preprocess_config.preprocessor_file_path, obj=preprocessor)
            logging.info("Preprocessing object saved")

            return train_arr, test_arr, self.preprocess_config.preprocessor_file_path
        
        
        except Exception as e:
            raise CustomException(e ,sys)