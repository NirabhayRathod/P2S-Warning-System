import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation 
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer1 import ClassificationTrainer


@dataclass
class Data_Path:
    raw_data_cl: str   = os.path.join('Artifacts', 'raw_data_cl.csv')
    train_data_cl: str = os.path.join('Artifacts', 'train_data_cl.csv')
    test_data_cl: str  = os.path.join('Artifacts', 'test_data_cl.csv')

    raw_data_rg :str=os.path.join('Artifacts' , 'raw_data_rg.csv')
    train_data_rg :str=os.path.join('Artifacts' , 'train_data_rg.csv')
    test_data_rg :str=os.path.join('Artifacts' , 'test_data_rg.csv')
    
class Ingestion:
    def __init__(self):
        self.data_path = Data_Path()
    
    def initiate_data_ingestion(self):
        logging.info('We are Entering in the data Ingestion Part ')
        try:
            data = pd.read_csv(os.path.join('dataset.csv'))
            logging.info('data reading successfully from a csv file')
            
            os.makedirs(os.path.dirname(self.data_path.raw_data_cl), exist_ok=True)
            data.to_csv(self.data_path.raw_data_cl, index=False, header=True)
            
            logging.info('We are doing--> train-test-split')
            train, test = train_test_split(data, test_size=0.2, random_state=42)
            
            logging.info('train and test data saved for classification')
            train.to_csv(self.data_path.train_data_cl, index=False, header=True)
            test.to_csv(self.data_path.test_data_cl, index=False, header=True)
            
            
            data=data[data['p_wave_detected']==1]
            train, test = train_test_split(data, test_size=0.2, random_state=42)
            train.to_csv(self.data_path.train_data_rg , index=False , header=True)
            test.to_csv(self.data_path.test_data_rg , index=False , header=True)
            logging.info('train and test data saved for regressiom')
            
            return (
                self.data_path.train_data_cl, 
                self.data_path.test_data_cl, 
                self.data_path.train_data_rg, 
                self.data_path.test_data_rg
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    obj = Ingestion()
    train_data_cl, test_data_cl, train_data_rg, test_data_rg = obj.initiate_data_ingestion()

    transformation_object = DataTransformation()

    train_arr_cl, test_arr_cl, train_arr_rg, test_arr_rg = transformation_object.initiate_data_transformation(
        train_data_cl, test_data_cl, train_data_rg, test_data_rg
    )

    model_obj = ModelTrainer()
    reg_score = model_obj.initiate_model_training(train_arr_rg, test_arr_rg)
    print(f"Our best model score for regression (R2): {reg_score}")
    
    class_obj = ClassificationTrainer()
    class_score = class_obj.initiate_model_training(train_arr_cl, test_arr_cl)
    print(f'Best model score for classification (accuracy): {class_score}')

