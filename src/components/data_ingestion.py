import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

## Intitialize the Data Ingestion Configuration

@dataclass

class Dataingestionconfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')

## create a class for Data Ingestion

class Dataingestion:
    def __init__(self):
        self.ingestion_config = Dataingestionconfig

    def initiate_data_ingestion(self):
        logging.info('Data ingestion process has begun')
        try:
            df=pd.read_csv('D:\\Ineuron\\Machine_Learning\\Delivery_time_Prediction\\Dataset\\finalTrain.csv')
            logging.info('Dataset has been read succesfully as Pandas Dataframe')

            ## Saving the entire data after reading into one dump raw CSV file.
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Train Test Split done with 30 percent test size')
            train_set, test_set = train_test_split(df,test_size=0.30,random_state=30)

            ## Saving the train & test data into their designated paths after the split.
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion has been completed')

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)


        except Exception as e:
            logging.error('Exception has occured at Data ingestion stage')
            raise CustomException (e,sys)
        
if __name__=='__main__':
    obj=Dataingestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()