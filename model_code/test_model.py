from data_preprocessing import X, Y
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from tensorflow import keras
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import random


feature_extractor = load_model("model_saved\custom_cnn_feature_extractor_final.h5")
classification_model = joblib.load("model_saved\RF_model_final.pkl")

def accuracy(Y_pred, Y):
    return np.sum( Y_pred == Y) / Y.size

def test_accuracy(X, Y):
    features = feature_extractor.predict(X)
    Y_pred = classification_model.predict(features)
    return accuracy(Y_pred, Y), Y_pred

def predict(X):
    features = feature_extractor.predict(X)
    Y_pred = classification_model.predict(features)
    probability = np.max(classification_model.predict_proba(features), axis=1)

    return Y_pred, probability

def test_prediction(index, X, Y):
    
    current = X[index]
    label = Y[index]
    current_image = np.expand_dims(current, axis=0)

    predicted_label, probability = predict(current_image)
    
    encode_label = {0:"Benign", 1:"Malignant", 2:"Normal"}

    print(f"Predicted label: {encode_label[int(predicted_label)]}")
    print(f"Actual label: {encode_label[int(label)]}")
    print(f"Confidence level: {probability}%")
  
    
    plt.imshow(current)
    plt.title(encode_label[int(predicted_label)])
    plt.axis("off")
    plt.show()

acc, Y_pred = test_accuracy(X, Y)

print(f"Accuracy: {acc * 100}%")
print(classification_report(Y, Y_pred))

for i in range(5): 
    index = random.randint(1, 500)
    test_prediction(index, X, Y)
