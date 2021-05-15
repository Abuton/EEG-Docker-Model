from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

class Cancer(BaseModel):
	concavity_worst: float
	symmentry: float
	concavity_mean: float
	concave_points_mean: float
	perimeter_worse: float
	compactness_worst: float
	concave_points: float
	compactness_mean: float
	texture: float
	area: float


@app.post("/predict")
async def predict_patient_cancer_state(patient: Cancer):
	filename = "pickled_files/breast_cancer_ML_model.pkl"
	data = patient.dict()
	loaded_model = pickle.load(open(filename, 'rb'))
	data_input = np.array([data['concavity_worst'], data['symmentry'], data['concavity_mean'], data['concave_points_mean'],
				data['perimeter_worse'], data['compactness_worst'], data['concave_points'], data['compactness_mean'],
				 data['texture'], data['area']])	

	data_input = data_input.reshape(1,-1)
	prediction = int(loaded_model.predict(data_input))
	if prediction == 0: prediction='Benign'
	else: prediction="Malignant"
	prediction_probability = loaded_model.predict_proba(data_input).max()

	return {
		"prediction": prediction,
		"probability": prediction_probability
	}

# if __name__ == '__main__':
# 	predict_patient_cancer_state()