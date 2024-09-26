# NYC Taxi Duration Prediction

This project predicts the duration of NYC taxi rides using machine learning models. It includes scripts for training the model and serving predictions via a Flask API.

## Setup

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/nyc-taxi-duration-prediction.git
   cd nyc-taxi-duration-prediction

   ```

2. **Create a virtual environment and activate it:**

   ```sh
   python3 -m venv venv
   source venv/bin/activate

   ```

3. **Install the required packages:**

   ```sh
   pip install -r requirements.txt

   ```

4. **Set up environment variables: Create a .env file in the root directory and add the following:**
   MODEL_VERSION=
   MODEL_URI=file:///path/to/your/model

## Training the Model

Run the train.py script to train the model:

````sh
python train.py

## Serving Predictions:
```sh
python train.py

## Testing the API
 Use the following curl command to test the /predict endpoint:

```sh
curl -X POST \
    -d '{
        "ride_id": "ride_xyz",
        "ride": {
            "PULocationID": 100,
            "DOLocationID": 102,
            "trip_distance": 30
        }
    }' \
    -H "Content-Type: application/json" \
    http://localhost:9696/predict

## Monitoring



```sh
{
    "request": {
        "PULocationID": 100,
        "DOLocationID": 102,
        "trip_distance": 30
    },
    "response": {
        "ride_id": "ride_xyz",
        "prediction": {
            "duration": 7.231998825001949
        },
        "model_version": "8a3db1cad7724bc5a9c1bd08ae3530eb"
    }
}



````
