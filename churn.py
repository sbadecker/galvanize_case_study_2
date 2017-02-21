import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

def data_wrangling(df):
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['days_since_last_trip'] = (dt.datetime.strptime('2014-07-01', "%Y-%m-%d") - df['last_trip_date']).dt.days
    df['churned'] = df.last_trip_date < '2014-06-01'
    df = pd.get_dummies(df, columns=['city', 'phone'])
    return df

if __name__ == '__main__':
    df = pd.read_csv('data/churn_train.csv')
    df = missing_data(df)
    df = data_wrangling(df)

    cols = [u'avg_dist', u'avg_rating_by_driver', u'avg_rating_of_driver',
       u'surge_pct',
       u'trips_in_first_30_days', u'luxury_car_user', u'weekday_pct', u'city_Astapor', u"city_King's Landing",
       u'phone_Android']

    y = df['churned']
    X = df[cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rand_forest = GradientBoostingClassifier()
    rand_forest.fit(X_train, y_train)
    print rand_forest.oob_score_
    print rand_forest.score(X_test, y_test)
    for i in range(len(cols)):
        print cols[i], rand_forest.feature_importances_[i]

    # log_model = LogisticRegressionCV()
    # log_model.fit(X_train, y_train)
    # print log_model.score(X_test, y_test)
    # for i in range(len(cols)):
    #     print cols[i], log_model.coef_[0][i]
