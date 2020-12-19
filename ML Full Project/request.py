import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Age':2, 'Glucose':9, 'BMI':6, 'DiabetesPedigreeFunction': 6 ,
	'Pregnancies':66 ,'SkinThickness': 66,'Insulin':55,
            'Glucose':55})

print(r.json())