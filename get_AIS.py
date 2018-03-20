import requests
import psycopg2
import datetime,time

USER = 'iceml'
PASSWORD = 'WinterNavigation'
HOST = 'localhost'
PORT = 5432
DB_NAME = 'iceml'
TABLE_NAME = 'ais_observation'
COLUMN_NAMES = ','.join(['timestamp', 'mmsi', 'location', 'sog', 'cog', 'navstat', 'posacc', 'raim', 'heading', 'timestamp_seconds'])

def connect():
    try:
        return psycopg2.connect(host=HOST,port=PORT,database=DB_NAME, user=USER, password=PASSWORD)
    except Exception as e:
        print("Db connection failed: {}".format(e))

def timeString(timeStamp):
    return timeStamp.replace(tzinfo=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]+'Z'

def geographyPoint(lon, lat): 
    return "ST_GeogFromText('SRID=4326;POINT({} {})')".format(lon, lat)

def getData(requestTime):
    location_url = 'https://meri.digitraffic.fi/api/v1/locations/latitude/63.3038413/longitude/20.7441543/radius/500/from/' + timeString(requestTime)
    print("GET: url={}".format(location_url))
    response = requests.get(url=location_url)
    data = response.json()
    print("GET Result: code={} type={} featuresLength={}".format(response.status_code, data['type'], len(data['features'])))
    return data

def writeFeature(feature, cursor):
    mmsi = feature["mmsi"]
    lon = feature["geometry"]["coordinates"][0]
    lat = feature["geometry"]["coordinates"][1]
    sog = feature["properties"]["sog"]
    cog = feature["properties"]["cog"]
    navStat = feature["properties"]["navStat"]
    posAcc = 1 if feature["properties"]["posAcc"] == "True" else 0
    raim = 1 if feature["properties"]["raim"] == "True" else 0
    heading = feature["properties"]["heading"]
    timestampSeconds = feature["properties"]["timestamp"]
    timeStamp = datetime.datetime.utcfromtimestamp(feature["properties"]["timestampExternal"]/1000.0)
    if lat >= 61:
        sql = "insert into {} ({}) values ('{}', {}, {}, {}, {}, {}, B'{}', B'{}', {}, {})".format(TABLE_NAME, COLUMN_NAMES, timeString(timeStamp), mmsi, geographyPoint(lon,lat), sog, cog, navStat, posAcc, raim, heading, timestampSeconds)
        cursor.execute(sql)
        return (timeStamp, True)
    else: 
        return (timeStamp, False)

def doRequest(requestTime, conn):
    with conn.cursor() as cursor:
        data = getData(requestTime)
        results = list(map(lambda d: writeFeature(d, cursor), data["features"]))
        maxTimeStamp = max(map(lambda r: r[0], results))
        writeCount = sum(1 for r in results if r[1])
        conn.commit()
        print("Wrote positions: handled={} wrote={} newest={}".format(len(data['features']), writeCount, timeString(maxTimeStamp)))
        return maxTimeStamp

def runOnce(requestTime, conn):
    maxWrittenTime = doRequest(requestTime, conn)
    print("Handled from {} to {}".format(requestTime, maxWrittenTime))
    return maxWrittenTime if maxWrittenTime > requestTime else requestTime

def run( requestTime, sleepSeconds, conn):
    while True:
        requestTime = runOnce(requestTime, conn)
        time.sleep(sleepSeconds)

if __name__ == "__main__":
    requestTime = datetime.datetime.utcnow() - datetime.timedelta(minutes=30)
    with connect() as conn:
        #runOnce(requestTime, conn)
        run(requestTime, 60, conn)