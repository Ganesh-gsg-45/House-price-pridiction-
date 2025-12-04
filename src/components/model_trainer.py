import os
import sys
import time
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV


from src.exception import CustomException
from src.loger import logging

from src.utils import save_object,evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            start_time = time.time()
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
        
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            logging.info("Starting selective model evaluation")
            eval_start = time.time( )
            model_report = {}
            best_model_score = 0
            best_model_name = None
            best_model = None
            model_order = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBRegressor", "CatBoosting Regressor", "AdaBoost Regressor"]

            for model_name in model_order:
                if model_name in models:
                    logging.info(f"Training {model_name}")
                    model_start = time.time()
                    # Train single model
                    model = models[model_name]
                    para = params[model_name]
                    gs = RandomizedSearchCV(model, para, cv=3, n_iter=10, random_state=42, n_jobs=-1)
                    gs.fit(X_train, y_train)
                    model.set_params(**gs.best_params_)
                    model.fit(X_train, y_train)
                    y_test_pred = model.predict(X_test)
                    test_model_score = r2_score(y_test, y_test_pred)
                    model_report[model_name] = test_model_score
                    model_end = time.time()
                    logging.info(f"{model_name} trained in {model_end - model_start:.2f} seconds with score: {test_model_score}")

                    # Check if good enough (e.g., > 0.8), select and stop
                    if test_model_score > 0.8:
                        best_model_score = test_model_score
                        best_model_name = model_name
                        best_model = model
                        logging.info(f"Selected {model_name} as best model early")
                        break
            eval_end = time.time()
            logging.info(f"Model evaluation completed in {eval_end - eval_start:.2f} seconds")
            
            # If no early selection, get the best from trained models
            if best_model is None:
                best_model_score = max(sorted(model_report.values()))
                best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
                best_model = models[best_model_name]

            if best_model_score < 0.0:
                logging.warning(f"Model score is very low: {best_model_score}. Using best available model anyway.")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            end_time = time.time()
            logging.info(f"Model training completed in {end_time - start_time:.2f} seconds with R2 score: {r2_square}")
            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
            

            
