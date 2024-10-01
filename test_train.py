import pytest
from train import train_rf, predict
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
import numpy as np


@pytest.fixture
def model_file():
    filename = 'rf_model.pkl'
    yield filename  # Fournit le nom du fichier du modèle


def test_model_file_exists(model_file):
    assert os.path.exists(model_file) 

@pytest.fixture
def loaded_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    yield model # Fournit le nmodèle chargé

def test_train(loaded_model):
    assert isinstance(loaded_model, RandomForestClassifier)


def test_predict(loaded_model):
    input_data = np.array([[1, 1, 1, 1]])
    assert input_data.shape[1] == 4
    prediction = predict(loaded_model, input_data)
    assert prediction[0] == 0



 
  
   

if __name__ == "__main__":
    pytest.main()
