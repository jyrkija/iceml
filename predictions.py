
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import StandardScaler


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


def timeString(timeStamp):
    return timeStamp.replace(tzinfo=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+'Z'

def readDataframe(columns, condition=None):
    engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(USER,PASSWORD,HOST,PORT,DB_NAME))
    where = '' if condition==None else 'where ' + condition
    sql = 'select {} from {} {} ORDER BY {}'.format(','.join(columns), TABLE_NAME, where, ORDER)
    print(sql)
    return pd.read_sql_query(sql,con=engine)

def buildRandomForestRegressor(X_train, y_train):
    regressor = RandomForestRegressor(n_estimators=500, random_state=0)
    regressor.fit(X_train,y_train)
    return regressor

def featureScaling(X_train, X_test, y_train):
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    return (X_train, X_test, y_train)

def evaluateRegressor(name, y_pred, y_test):
    print("{} Predictions {}: MeanSquaredError={} ExplainedVariance={}".format(name, len(y_test), mean_squared_error(y_test, y_pred), explained_variance_score(y_test, y_pred)))

if __name__ == "__main__":
    #allData = readDataframe(ALL_COLUMNS)
    #time12hAgo = datetime.datetime.utcnow() - datetime.timedelta(hours=12)
    #last12hData = readDataframe(ALL_COLUMNS, "timestamp>'{}'".format(timeString(time12hAgo)))
    #oneShip = readDataframe(ALL_COLUMNS, "mmsi=230938570")
    trainData = readDataframe(ALL_COLUMNS, "MOD(mmsi,4)>0").dropna()
    testData = readDataframe(ALL_COLUMNS, "MOD(mmsi,4)=0").dropna()
    X_train = trainData.iloc[:,[3,4,6,7,9]].values
    X_test = testData.iloc[:,[3,4,6,7,9]].values
    y_train = trainData.iloc[:,10].values
    y_test = testData.iloc[:,10].values
    
    # Predict ships' future speed
    regressor = buildRandomForestRegressor(X_train, y_train)
    y_pred = regressor.predict(X_test)
    evaluateRegressor("Random Forest", y_pred, y_test)
    
