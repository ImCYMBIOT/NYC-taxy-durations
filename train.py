#!/usr/bin/env python
# coding: utf-8


from datetime import datetime

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
from sklearn.pipeline import make_pipeline

# Set the tracking URI to your MLFlow server
mlflow.set_tracking_uri("http://localhost:5000")

# Set the experiment name
mlflow.set_experiment("nyc-taxi-experiment-01")

artifact_location = "./mlruns"
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

def run(train_month: datetime, val_month= datetime ):
    template_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year:04d}-{month:02d}.parquet'
    train_url = template_url.format(year = train_month.year, month = train_month.month)
    val_url = template_url.format(year = val_month.year, month = val_month.month)
    
    df_train = read_dataframe(train_url)
    df_val = read_dataframe(val_url)


    with mlflow.start_run():
        categorical = ['PULocationID', 'DOLocationID']
        numerical = ['trip_distance']


        mlflow.log_params({
            'categorical': categorical,
            'numerical': numerical,
            'train_month': train_month,
            'val_month': val_month
        })
        model_params = dict(
            fit_intercept = False
        )
        mlflow.log_params(model_params)
        
        pipeline = make_pipeline(
            DictVectorizer(),
            LinearRegression(**model_params)
            
        )
        
        target = 'duration'
        y_train = df_train[target].values
        y_val = df_val[target].values

        train_dicts = df_train[categorical + numerical].to_dict(orient='records')
        pipeline.fit(train_dicts, y_train) 
        
        val_dicts = df_val[categorical + numerical].to_dict(orient='records')   
        y_pred = pipeline.predict(val_dicts)
        
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(rmse)
        mlflow.log_metric("rmse", rmse)

        mlflow.sklearn.log_model(pipeline, 'model')
        
def main():

    run(
        train_month = datetime(year = 2022, month = 1, day = 1),
        val_month = datetime(year = 2022, month = 2, day = 1)
    )


if __name__ ==  "__main__":
    main()