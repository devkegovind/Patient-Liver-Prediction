import os
import sys
import pandas as pd 
import numpy as np 

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            logging.info("Exception occurs in Prediction")
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self,
                 Age:int,
                Gender:str,
                Total_Bilirubin: float,
                Direct_Bilirubin:float,
                Alkaline_Phosphotase: int,
                Alamine_Aminotransferase: int,
                Aspartate_Aminotransferase: int,
                Total_Protiens: float,
                Albumin: float,
                Albumin_and_Globulin_Ratio: float
                ):
                self.Age = Age
                self.Gender = Gender
                self.Total_Bilirubin = Total_Bilirubin
                self.Direct_Bilirubin = Direct_Bilirubin
                self.Alkaline_Phosphotase = Alkaline_Phosphotase
                self.Alamine_Aminotransferase = Alamine_Aminotransferase
                self.Aspartate_Aminotransferase = Aspartate_Aminotransferase
                self.Total_Protiens = Total_Protiens
                self.Albumin = Albumin
                self.Albumin_and_Globulin_Ratio = Albumin_and_Globulin_Ratio


    def get_data_as_dataframe(self):
         try:
              custom_data_input_dict = {
                   'Age':[self.Age],
                   'Gender':[self.Gender],
                   'Total_Bilirubin':[self.Total_Bilirubin],
                   'Direct_Bilirubin':[self.Direct_Bilirubin],
                   'Alkaline_Phosphotase':[self.Alkaline_Phosphotase],
                    'Alamine_Aminotransferase' : [self.Alamine_Aminotransferase],
                    'Aspartate_Aminotransferase':[self.Aspartate_Aminotransferase],
                    'Total_Protiens':[self.Total_Protiens],
                    'Albumin':[self.Albumin],
                    'Albumin_and_Globulin_Ratio':[self.Albumin_and_Globulin_Ratio]
                    
              }
              df = pd.DataFrame(custom_data_input_dict)
              logging.info("Dataframe Gathered")
              return df
         except Exception as e:
              logging.info("Exception occurs in Prediction Pipeline")
              raise CustomException(e, sys)


