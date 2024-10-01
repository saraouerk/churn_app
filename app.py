from flask import Flask, request, jsonify
import numpy as np
import pickle
import sys
  
app = Flask(__name__) # Initialize the flask App

# Chemin du fichier du modèle
MODEL_FILE = 'rf_model.pkl'

# Charger le modèle
def load_model():
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model



@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json.get('input_data')  # Récupérer les données d'entrée du corps de la requête
    if input_data is None:
        return jsonify({"error": "No input data provided!"}), 400

    model = load_model()
    input_array = np.array(input_data).reshape(1, -1)
    #return (jsonify({"input_array": input_array[0][0]}))
    prediction = model.predict(input_array)

    return jsonify({"prediction": prediction[0].item()})
    #return prediction


if __name__ == '__main__':
    app.run(debug=True)
