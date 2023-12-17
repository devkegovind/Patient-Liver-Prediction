import os
import sys
import pickle
import numpy as np 
import pandas as pd 

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

from src.utils import save_object
from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging


@dataclass 

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")

            # Define which columns should be ordinal encoded
            cat_cols = ['Gender']

            num_cols = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase','Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']

            # Define the custom ranking for each ordinal variables
            Gender_cat = ['Male', 'Female']

            logging.info("Pipeline Initiated")

            # Numerical Pipeline

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[Gender_cat])),
                    ('scaler', StandardScaler())

                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols)
                ]
            )
            logging.info("Pipeline Completed")

            return preprocessor
        
        except Exception as e:
            logging.info("Exception occurs in the Data Transformation")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:

            "Reading Train & Test Data"

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test Data Completed")
            print()
            logging.info(f"Shape of Train Dataframe : {train_df.shape}")
            print()
            logging.info(f"Train Dataframe Head:\n{train_df.head().to_string()}")
            print()
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")
            print()
            logging.info(f"Shape of Test Dataframe : {test_df.shape}")
            print()
            logging.info(f"Train Dataframe Tail : \n{train_df.tail().to_string()}")
            print()
            logging.info(f"Testing Dataframe Tail : \n{test_df.tail().to_string()}")
            print()
            logging.info("Obtaining Preprocessing Object")

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Dataset'

            input_feature_train_df = train_df.drop(columns = target_column_name, axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = target_column_name, axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Input_Feature_Train_df : \n {input_feature_train_df}")
            print()
            logging.info(f"Target_feature_Train_df : \n{target_feature_train_df}")
            print()
            logging.info(f"Input_Feature_Test_df : \n{input_feature_test_df}")
            print()
            logging.info(f"Target_Feature_Test_df : \n{target_feature_test_df}")
            print()
            logging.info(f"Input Feature Train df shape : {input_feature_train_df.shape}")
            print()
            logging.info(f"Target Feature Train df shape : {target_feature_train_df.shape}")
            print()
            logging.info(f"Input Feature Test df shape : {input_feature_test_df.shape}")
            print()
            logging.info(f"Target Feature Test df shape : {target_feature_test_df.shape}")
            print()

            # Transforming Using Preprocessing Object
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Input_feature_train_arr : \n {input_feature_train_arr}")
            print()
            logging.info(f"Input_Feature_Test_Arr :\n{input_feature_test_arr}")
            print()
            logging.info("Applying Preprocessing object on Training & Testing Dataset")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Train_arr :\n{train_arr}")
            print()
            logging.info(f"Test arr : \n{test_arr}")
            print()
            logging.info(f"Train Arr shape :{train_arr.shape}")
            print()
            logging.info(f"Test arr shape : {test_arr.shape}")
            print()

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Preprocessor Pickle File saved")
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            logging.info("Exception occurs in the initiate_data transformation")
            raise CustomException(e, sys)



if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)






