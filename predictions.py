
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sqlalchemy import create_engine

USER = 'iceml'
PASSWORD = 'WinterNavigation'
DB_URL = 'localhost:5432'
DB_NAME = 'iceml'
ALL_COLUMNS = ['id', 'timestamp', 'mmsi', 
               'ST_Y(location::geometry) lat', 'ST_X(location::geometry) lon',
               'ST_AsText(location::geometry) location_str',
               'sog', 'cog', 'heading',
               'navstat', 'posacc', 'raim']
TABLE_NAME = 'ais_observation'


def timeString(timeStamp):
    return timeStamp.replace(tzinfo=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+'Z'

def readDataframe(columns, condition=None):
    engine = create_engine('postgresql://{}:{}@{}/{}'.format(USER,PASSWORD,DB_URL,DB_NAME))
    where = '' if condition==None else 'where ' + condition
    sql = 'select {} from {} {}'.format(','.join(columns),TABLE_NAME, where)
    print(sql)
    return pd.read_sql_query(sql,con=engine)

#allData = readDataframe(ALL_COLUMNS)
#time12hAgo = datetime.datetime.utcnow() - datetime.timedelta(hours=12)
#last12hData = readDataframe(ALL_COLUMNS, "timestamp>'{}'".format(timeString(time12hAgo)))
#oneShip = readDataframe(ALL_COLUMNS, "mmsi=230938570")
trainData = readDataframe(ALL_COLUMNS, "MOD(mmsi,4)>0")
testData = readDataframe(ALL_COLUMNS, "MOD(mmsi,4)=0")
X_train = trainData.iloc[:,[3,4,7]].values
X_test = testData.iloc[:,[3,4,7]].values
y_train = trainData.iloc[:,6].values
y_test = testData.iloc[:,6].values

# Predict ships stopping

"""
# Get use location and course to predict speed
X = allData.iloc[:,[3,4,7]].values
y = allData.iloc[:,6].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
"""

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
"""

# Random forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train,y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error, explained_variance_score
meanSquared = mean_squared_error(y_test, y_pred)
explainedVariance = explained_variance_score(y_test, y_pred)

# 



