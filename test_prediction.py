"""
Test script to make predictions using the trained model
"""

import sys
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.loger import logging
from src.exception import CustomException

def main():
    try:
        logging.info("="*50)
        logging.info("STARTING PREDICTION PIPELINE")
        logging.info("="*50)
        
        # Create sample data for prediction
        logging.info("\nCreating sample house data...")
        sample_data = CustomData(
            bedrooms=3,
            bathrooms=2.0,
            sqft_living=2000,
            sqft_lot=5000,
            floors=2,
            waterfront=0,
            view=0,
            condition=3,
            sqft_above=1500,
            sqft_basement=500,
            yr_built=2000,
            yr_renovated=0
        )
        
        # Convert to dataframe
        data_df = sample_data.get_data_as_dataframe()
        logging.info(f"Input data:\n{data_df}")
        
        # Make prediction
        logging.info("\nLoading model and making prediction...")
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(data_df)
        
        logging.info(f"\n{'='*50}")
        logging.info(f"PREDICTION RESULT")
        logging.info(f"{'='*50}")
        logging.info(f"Predicted House Price: ${prediction[0]:,.2f}")
        logging.info(f"{'='*50}\n")
        
        print(f"\n✓ Predicted Price: ${prediction[0]:,.2f}")
        
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
