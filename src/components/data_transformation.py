import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.loger import logging
import os
from src.utils import save_object
@dataclass
class DataTransformationconfig:
    preprocessor_obj_path=os.path.join('artifacts','preprocess.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
    def get_data_transformation_object(self):
        try:
            df=pd.read_csv('C:\\Users\\T Ganesh\\OneDrive\\Desktop\\gsg\\housepridiction\\data\\data.csv')
            df.drop(['date','street', 'city',
             'statezip', 'country'],axis=1,inplace=True)
            numerical_cols=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
             'yr_built', 'yr_renovated']
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info(f"Numerical columns: {numerical_cols}")  
            preprocessor=ColumnTransformer(
                [('num_pipeline',num_pipeline,numerical_cols)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            if os.path.exists(self.data_transformation_config.preprocessor_obj_path):
                logging.info("Preprocessor already exists, loading cached object")
                from src.utils import load_object
                preprocessing_obj = load_object(file_path=self.data_transformation_config.preprocessor_obj_path)
            else:
                logging.info("Obtaining preprocessing object")
                preprocessing_obj=self.get_data_transformation_object()
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            target_column_name='price'
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            #only fit it not cached
            if not os.path.exists(self.data_transformation_config.preprocessor_obj_path):
                input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            else:
                input_feature_train_arr=preprocessing_obj.transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]  
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]      
            logging.info(f"Saved preprocessing object.")  
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            ) 
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
