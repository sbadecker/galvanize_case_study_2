import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
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

    df_test = pd.read_csv('data/churn_test.csv')
    df_test = missing_data(df_test)
    df_test = data_wrangling(df_test)

    cols = [u'avg_dist', u'avg_rating_by_driver', u'avg_rating_of_driver',
       u'surge_pct',
       u'trips_in_first_30_days', u'luxury_car_user', u'weekday_pct', u'city_Astapor', u"city_King's Landing",
       u'phone_Android',u'churned']

    #re-added in 'churned' column for comparison_test, but drop it in line 45 and 46
    y = df['churned']
    X = df[cols]

    y_test = df_test['churned']
    X_test = df_test[cols]

    #added comparison_test to check our predictions vs actual churned
    comparison_test = X_test.copy()
    X.drop('churned',axis = 1,inplace=True)
    X_test.drop('churned',axis = 1, inplace=True)
    cols.remove('churned')
    grad = GradientBoostingClassifier()
    grad.fit(X, y)
    print grad.score(X_test, y_test)
    print confusion_matrix(y_test, grad.predict(X_test))


    for i in range(len(cols)):
         print cols[i], grad.feature_importances_[i]

    #churn_prediction is the result/predicted churn using our model
    comparison_test['churn_prediction'] = grad.predict(X_test)
    #correct_prediction returns True for correct prediction compared to actual churn
    comparison_test['correct_prediction']= comparison_test['churned']==comparison_test['churn_prediction']

    #Created two DFs to access the correct and false predictions easily
    correct_predict = comparison_test[comparison_test['correct_prediction']==True]
    false_predict = comparison_test[comparison_test['correct_prediction']==False]

    # correct_predict.hist(figsize=(10,10), bins=25)
    # false_predict.hist(figsize=(10,10), bins=25)
    # plt.show()
    
    '''
    # log_model = LogisticRegressionCV()
    # log_model.fit(X_train, y_train)
    # print log_model.score(X_test, y_test)
    # for i in range(len(cols)):
    #     print cols[i], log_model.coef_[0][i]
    '''
