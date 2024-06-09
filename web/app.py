# For this to work, you must install: pip install scikit-learn==1.2.2
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model 
from image_preprocess import preprocess
import os
import numpy as np
import joblib

feature_extractor = load_model("\\final_model\\custom_cnn_feature_extractor_finale.h5")
classification_model = joblib.load("\\final_model\\RF_model_finale.pkl")

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['GET'])

def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])

def predict():
    if request.method == 'POST':
        imagefile = request.files['imagefile']
        image_path = os.path.join(app.root_path, 'static\images', imagefile.filename)
        imagefile.save(image_path)

        X = preprocess(image_path)
        encode_label = {0:"Benign", 1:"Malignant", 2:"Normal"}

        features = feature_extractor.predict(X)
        features = features.reshape(features.shape[0], -1)

        Y_pred = classification_model.predict(features)
        probability = np.max(classification_model.predict_proba(features), axis=1)*100
        classification = '%s (%.2f%%)' % (encode_label[int(Y_pred)], probability)

        return render_template('index.html', prediction=classification, image=imagefile.filename)

if __name__ == '__main__':
    app.run(port=3000, debug =True)

