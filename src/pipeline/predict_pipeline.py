import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, save_object
from src.components.data_transformation import DataTransformation
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            # Resolve artifacts directory relative to project root (works when deployed)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            # prefer deploy_artifacts when present (useful for deployments that include packaged artifacts)
            deploy_dir = os.path.join(project_root, 'deploy_artifacts')
            artifacts_dir = os.path.join(project_root, 'artifacts')
            if os.path.exists(deploy_dir):
                artifacts_dir = deploy_dir
            model_path = os.path.join(artifacts_dir, 'model.pkl')
            preprocessor_path = os.path.join(artifacts_dir, 'preprocess.pkl')
            print(f"Loading model from: {model_path}")
            print(f"Loading preprocessor from: {preprocessor_path}")

            if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Missing artifact files. Expected: {model_path} and {preprocessor_path}")

            model = load_object(file_path=model_path)
            try:
                preprocessor = load_object(file_path=preprocessor_path)
            except Exception as e:
                # Common cause: sklearn version mismatch (pickled classes moved/renamed)
                print(f"Failed to load preprocessor pickle: {e}")
                print("Attempting to rebuild preprocessor from code as a fallback...")
                # Rebuild preprocessor object using the same transformation logic
                data_transformer = DataTransformation()
                preprocessor = data_transformer.get_data_transformation_object()
                # Save rebuilt preprocessor so future loads succeed
                try:
                    save_object(file_path=preprocessor_path, obj=preprocessor)
                    print(f"Saved rebuilt preprocessor to: {preprocessor_path}")
                except Exception as s_exc:
                    print(f"Warning: failed to save rebuilt preprocessor: {s_exc}")
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
