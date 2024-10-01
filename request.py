import requests

# URL de l'API
base_url = 'http://127.0.0.1:5000'



# Faire une prédiction
input_data = {"input_data": [1, 1, 1, 1]}
predict_response = requests.post(f'{base_url}/predict', json=input_data)

# print('Predict Response Status Code:', predict_response.status_code)  
  
# if predict_response.ok:
print('Predict Response:', predict_response.json())
# else:
#     print('Failed to make prediction')