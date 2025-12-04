import streamlit as st
import pandas as pd
import traceback
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException


st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("House Price Prediction")
st.write("Enter the house features below and click Predict.")

with st.form(key='predict_form'):
	bedrooms = st.number_input('Bedrooms', min_value=0, max_value=10, value=3, step=1)
	bathrooms = st.number_input('Bathrooms', min_value=0.0, max_value=10.0, value=2.0, step=0.25)
	sqft_living = st.number_input('Sqft Living', min_value=100, max_value=20000, value=1500, step=50)
	sqft_lot = st.number_input('Sqft Lot', min_value=100, max_value=100000, value=5000, step=50)
	floors = st.number_input('Floors', min_value=1.0, max_value=4.0, value=1.0, step=0.5)
	waterfront = st.selectbox('Waterfront (0 = no, 1 = yes)', options=[0,1], index=0)
	view = st.number_input('View (0-4)', min_value=0, max_value=4, value=0, step=1)
	condition = st.number_input('Condition (1-5)', min_value=1, max_value=5, value=3, step=1)
	sqft_above = st.number_input('Sqft Above', min_value=0, max_value=20000, value=1200, step=50)
	sqft_basement = st.number_input('Sqft Basement', min_value=0, max_value=10000, value=300, step=50)
	yr_built = st.number_input('Year Built', min_value=1800, max_value=2025, value=1990, step=1)
	yr_renovated = st.number_input('Year Renovated (0 if none)', min_value=0, max_value=2025, value=0, step=1)

	submit = st.form_submit_button('Predict')

if submit:
	try:
		custom = CustomData(
			bedrooms=int(bedrooms),
			bathrooms=float(bathrooms),
			sqft_living=int(sqft_living),
			sqft_lot=int(sqft_lot),
			floors=float(floors),
			waterfront=int(waterfront),
			view=int(view),
			condition=int(condition),
			sqft_above=int(sqft_above),
			sqft_basement=int(sqft_basement),
			yr_built=int(yr_built),
			yr_renovated=int(yr_renovated)
		)

		input_df = custom.get_data_as_dataframe()
		pipeline = PredictPipeline()
		prediction = pipeline.predict(input_df)
		st.success(f"Predicted Price: ${prediction[0]:,.2f}")
	except CustomException as ce:
		st.error(f"Prediction failed: {ce}")
		st.text(traceback.format_exc())
	except Exception as e:
		st.error(f"An unexpected error occurred: {e}")
		st.text(traceback.format_exc())

