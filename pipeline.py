"""
Main Pipeline to run the complete ML workflow:
1. Data Ingestion
2. Data Transformation
3. Model Training
"""

import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.loger import logging
from src.exception import CustomException

def main():
    try:
        logging.info("="*50)
        logging.info("STARTING ML PIPELINE")
        logging.info("="*50)
        
        # Step 1: Data Ingestion
        logging.info("\n>>> STEP 1: DATA INGESTION")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed")
        logging.info(f"Train data: {train_data_path}")
        logging.info(f"Test data: {test_data_path}")
        
        # Step 2: Data Transformation
        logging.info("\n>>> STEP 2: DATA TRANSFORMATION")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info(f"Data transformation completed")
        logging.info(f"Train array shape: {train_arr.shape}")
        logging.info(f"Test array shape: {test_arr.shape}")
        
        # Step 3: Model Training
        logging.info("\n>>> STEP 3: MODEL TRAINING")
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model training completed with R² Score: {r2_score:.6f}")
        
        logging.info("\n" + "="*50)
        logging.info("ML PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("="*50)
        
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
