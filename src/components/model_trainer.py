import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

from dataclasses import dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Dependent & Independent Variables From Dataset")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "LogisticRegression" : LogisticRegression(),
                "DecisionTreeClassifier" : DecisionTreeClassifier(),
                "RandomForestClassifier" : RandomForestClassifier(),
                "SupportVectorClassifier" : SVC()
            }

            model_report : dict = evaluate_models(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print()

            logging.info(f"Model Report : {model_report}")

            # To get Model Score From Dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name : {best_model_name}, Accuracy_Score : {best_model_score}")
            print()
            logging.info(f"Best Model Found, Model Name : {best_model_name}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )


        except Exception as e:
            logging.info("Exception occurs at Model Trainer")
            raise CustomException(e, sys)
        


