import os
from dotenv import load_dotenv
import mlflow
from flask import Flask, request, jsonify
import json

# Load environment variables from .env file
load_dotenv()

MODEL_VERSION = os.getenv('MODEL_VERSION')
MODEL_URI = os.getenv('MODEL_URI')

if MODEL_URI is None:
    raise ValueError("MODEL_URI environment variable is not set")

model = mlflow.pyfunc.load_model(MODEL_URI)

def prepare_features(ride):
    features = {}
    features['PULocationID'] = str(ride['PULocationID'])
    features['DOLocationID'] = str(ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    preds = model.predict(features)
    return float(preds[0])

app = Flask('duration-prediction')

def log(**kwargs):
    log_line = json.dumps(kwargs)
    with open('prediction_log', 'a') as f_out:
        f_out.write(log_line + '\n')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    ride_id = data['ride_id']
    ride = data['ride']

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'ride_id': ride_id,
        'prediction': {
            'duration': pred,
        },
        'model_version': MODEL_VERSION
    }
    log(request=ride, response=result)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)