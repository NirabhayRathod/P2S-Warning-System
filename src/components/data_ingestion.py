import os
import sys
from exception import CustomException
from logger import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class Data_Path:
    raw_data   :str=os.path.join('Artifact','raw_data.csv')
    train_data :str=os.path.join('Artifact','text_data.csv')
    test_data  :str=os.path.join('Artifacts','test.csv')

class Ingestion:
    def __init__(self):
        self.data_path=Data_Path()
    
    def initialize_data_ingestion(self):
        logging.info('We are Entering in the data Ingestion Part ')
        
        try:
            data=pd.read_csv('P2S-Warning-System\cleaned_pwave_ttf.csv')
            logging.info('data reading successfully from a csv file')
            
            os.makedirs(os.path.dirname(self.data_path.raw_data), exist_ok=True)
            data.to_csv(self.data_path.raw_data ,index=False , header=False)
            
            logging.info('We are doing--> train-test-split')
            train ,test=train_test_split(data, test_size=0.2 , random_state=42)
            
            train.to_csv(self.data_path.train_data ,index=False , header=False)
            test.to_csv(self.data_path.test_data , index=False , header=False )
            
            return(
                self.data_path.train_data , self.data_path.test_data
            )
            
        except Exception as e:
            raise CustomException(e , sys)