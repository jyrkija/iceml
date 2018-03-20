
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from copy import deepcopy


USER = 'iceml'
PASSWORD = 'WinterNavigation'
HOST = 'localhost'
PORT = 5432
DB_NAME = 'iceml'
ALL_COLUMNS = ['id', 'timestamp', 'mmsi', 
               'ST_Y(location::geometry) lat', 'ST_X(location::geometry) lon',
               'ST_AsText(location::geometry) location_str',
               'sog', 'cog', 'heading',
               'LAG(sog,5) OVER (PARTITION BY mmsi ORDER BY timestamp) prev_5_sog',
               'LEAD(sog,15) OVER (PARTITION BY mmsi ORDER BY timestamp) next_15_sog',
               'navstat', 'posacc', 'raim']
ORDER = 'timestamp'
TABLE_NAME = 'ais_observation'

    
def readDataframe(columns, condition=None):
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(USER,PASSWORD,HOST,PORT,DB_NAME))
    where = '' if condition==None else 'where ' + condition
    sql = 'select {} from {} {} ORDER BY {}'.format(','.join(columns), TABLE_NAME, where, ORDER)
    print(sql)
    return pd.read_sql_query(sql,con=engine)

class Features:
    def __init__(self, trainData, testData, x_columns, y_column):
        self.X_train = trainData.iloc[:,x_columns].values
        self.X_test = testData.iloc[:,x_columns].values
        self.y_train = trainData.iloc[:,y_column].values
        self.y_test = testData.iloc[:,y_column].values

class ScaledFeatures:
    def __init__(self, features):
        sc_X = StandardScaler()
        self.X_train = sc_X.fit_transform(features.X_train)
        self.X_test = sc_X.transform(features.X_test)
        sc_y = StandardScaler()
        self.y_train = sc_y.fit_transform(features.y_train.reshape(-1,1)).ravel()
        self.y_test = sc_y.transform(features.y_test.reshape(-1,1)).ravel()

def testRegressor(name, regressor, features):
    regressor.fit(features.X_train,features.y_train)
    y_pred = regressor.predict(features.X_test)
    msqe = mean_squared_error(features.y_test, y_pred)
    evs = explained_variance_score(features.y_test, y_pred)
    print("{} predictions {}: MeanSquaredError={} ExplainedVariance={}".format(name, len(y_pred), msqe, evs))

if __name__ == "__main__":
    trainData = readDataframe(ALL_COLUMNS, "MOD(mmsi,4)>0").dropna()
    testData = readDataframe(ALL_COLUMNS, "MOD(mmsi,4)=0").dropna()
    
    features = Features(trainData, testData, [3,4,6,7,9], 10)
    featuresSc = ScaledFeatures(features)
    
    # Predict ships' future speed - Random Forest
    testRegressor("Random Forest(500)", RandomForestRegressor(n_estimators=500, random_state=0), features)

    # Predict ships' future speed - SVR
    testRegressor("SVR(RBF)", SVR(kernel='rbf'), featuresSc)