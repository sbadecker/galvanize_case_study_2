import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
%matplotlib inline

def data_wrangling(df):
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['days_since_last_trip'] = (dt.datetime.strptime('2014-07-01', "%Y-%m-%d") - df['last_trip_date']).dt.days
    df.loc[df['days_since_last_trip'] > 30,'churned'] = 1
    df.loc[df['days_since_last_trip'] <= 30,'churned'] = 0
    return df

def missing_data(df):
    # Rating by drivers
    df['rated_by_driver'] = ~pd.isnull(df['avg_rating_by_driver'])
    df['avg_rating_by_driver'].fillna(-10, inplace=True)
    # Rating of drivers
    df['rated_driver'] = ~pd.isnull(df['avg_rating_of_driver'])
    df['avg_rating_of_driver'].fillna(-10, inplace=True)
    # Phone
    df['phone'].fillna('iPhone', inplace=True)
    return df

if __name__ == '__main__':
    df = pd.read_csv('data/churn_train.csv')
    df = data_wrangling(df)
    df = missing_data(df)
