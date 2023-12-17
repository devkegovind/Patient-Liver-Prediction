import os
import sys
import pandas as pd 
import numpy as np 

from src.logger import logging
from src.exception import CustomException

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:

    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Methods Start")

        try:

            # Import Dataset

            df = pd.read_csv(os.path.join("D:\\Downloads\\archive (3)\\indian_liver_patient.csv"))

            logging.info("Dataset Read as Pandas Dataframe")

            # Make Directory

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Create Path of Raw Data File Path
            df.to_csv(self.ingestion_config.raw_data_path, index = False)

            logging.info("Train Test Split")

            # Split The Dataframe into Train & Test
            train_set, test_set = train_test_split(df, test_size=.20, random_state=42)

            # Create Train & Test csv Files

            train_set.to_csv(self.ingestion_config.train_data_path, index = False)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False)

            logging.info("Data Ingestion is Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.info("Exception is occurs at Data Ingestion")
            raise CustomException(e, sys)
        



if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()