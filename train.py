import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import pickle 

data_path = "data"



def preprocessing(dataset):
    dataset_learn = dataset[['Age', 'Total_Purchase', 'Years',
       'Num_Sites', 'Churn']]
    X = dataset_learn[dataset_learn.columns[:-1]]
    y = dataset_learn[dataset_learn.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)
    return X_train, X_test, y_train, y_test

def train_rf(X_train, y_train):
    
    # class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    # weights_dict = dict(zip([0,1], class_weight))
    #print(weights_dict)
    #clf = RandomForestClassifier(random_state=0, class_weight = weights_dict)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, y_train)
    return clf

def predict(clf, X_test):
    return clf.predict(X_test)

if __name__ == "__main__":
    dataset = pd.read_csv(os.path.join(data_path, "customer_churn.csv"), parse_dates=['Onboard_date'])
    X_train, X_test, y_train, y_test = preprocessing(dataset)
    clf = train_rf(X_train, y_train)
    predictions = predict(clf, X_test)
    print(classification_report(y_test, predictions))
    filename = 'rf_model.pkl'
    pickle.dump(clf, open(filename, 'wb')) 
