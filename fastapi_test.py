import requests

patient_measurement = {
	'concavity_worst': 0.9,
	'symmentry': 0.3,
	'concavity_mean': 0.4,
	'concave_points_mean': 0.1,
	'perimeter_worse': 0.2,
	'compactness_worst': 0.7,
	'concave_points': 0.7,
	'compactness_mean': 0.09,
	'texture': 0.078,
	'area': 60
}

response = requests.post("http://127.0.0.1:8000/predict_patient_cancer_state_predict_post", json = patient_measurement)

print(response.content)