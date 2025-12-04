import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocess.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
class CustomData:
    def __init__( self,bedrooms:int, bathrooms:float, sqft_living:int, sqft_lot:int, floors:int,
       waterfront:int, view:int, condition:int, sqft_above:int, sqft_basement:int,
       yr_built:int, yr_renovated:int):
        self.bedrooms=bedrooms
        self.bathrooms=bathrooms
        self.sqft_living=sqft_living
        self.sqft_lot=sqft_lot
        self.floors=floors
        self.waterfront=waterfront
        self.view=view
        self.condition=condition
        self.sqft_above=sqft_above
        self.sqft_basement=sqft_basement
        self.yr_built=yr_built
        self.yr_renovated=yr_renovated
                 
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "bedrooms":[self.bedrooms],
                "bathrooms":[self.bathrooms],
                "sqft_living":[self.sqft_living],
                "sqft_lot":[self.sqft_lot],
                "floors":[self.floors],
                "waterfront":[self.waterfront],
                "view":[self.view],
                "condition":[self.condition],
                "sqft_above":[self.sqft_above],
                "sqft_basement":[self.sqft_basement],
                "yr_built":[self.yr_built],
                "yr_renovated":[self.yr_renovated]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
