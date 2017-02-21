import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def data_wrangling(df):
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['days_since_last_trip'] = (dt.datetime.strptime('2014-07-01', "%Y-%m-%d") - df['last_trip_date']).dt.days
    df.loc[df['days_since_last_trip'] > 30,'churned'] = True
    df.loc[df['days_since_last_trip'] <= 30,'churned'] = False
    df = pd.get_dummies(df, columns=['city', 'phone'])
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
    df = missing_data(df)
    df = data_wrangling(df)

    cols = ['avg_dist']
    # cols = [u'avg_dist', u'avg_rating_by_driver', u'avg_rating_of_driver',
    #    u'avg_surge', u'surge_pct',
    #    u'trips_in_first_30_days', u'luxury_car_user', u'weekday_pct',
    #    u'rated_by_driver', u'rated_driver', u'days_since_last_trip', u'city_Astapor', u"city_King's Landing", u'city_Winterfell',
    #    u'phone_Android', u'phone_iPhone']

    y = df['churned']
    X = df[cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rand_forest = RandomForestClassifier(oob_score=True)
    rand_forest.fit(X_train, y_train)
