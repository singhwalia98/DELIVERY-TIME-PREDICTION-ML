import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  
from src.utils import save_object 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation has started')

            ## Segregating the data into numerical & categorical columns.
            categorical_columns = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order',
                                    'Type_of_vehicle', 'Festival', 'City']

            numerical_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude', 
                                 'Restaurant_longitude', 'Delivery_location_latitude',
                                 'Delivery_location_longitude', 'Vehicle_condition',
                                'multiple_deliveries']


            logging.info('Pipeline initiated')
            ## Numerical pipeline
            num_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))]
            )

            ## Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                    ('One_Hot_Encoder', OneHotEncoder(categories='auto', handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))]
            )

            preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, numerical_columns),
            ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            
            logging.info('Pipeline completed')
            return preprocessor           

        except Exception as e:
            logging.error('Error in Data Transformation has occured')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info('Reading Train and Test data started')
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info('Reading Train and Test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name,'Delivery_person_ID', 'ID', 'Order_Date', 'Time_Orderd', 'Time_Order_picked']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Transforming using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            
            ## Converting it into a numpy array so we can load it in a much faster way. 
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                )
            
            logging.info('Preprocessing pickle file saved')

            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error('Exception occured in the initiate_data_transformation')
            raise CustomException(e,sys)
        
    