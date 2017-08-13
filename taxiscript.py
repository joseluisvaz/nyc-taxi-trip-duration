# this is a script for the kaggle competition

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from datetime import timedelta
import datetime as dt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

##converting everything to datetime

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime( test.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date']  = test['pickup_datetime'].dt.date

##change binary files into boolean

train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')
test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')

## creating a log_duration field by applying a log transform and normalizing the data

train['log_duration'] = np.log(train['trip_duration'].values + 1)
logmean = train['log_duration'].mean()
logstd = train['log_duration'].std()
train['log_duration'] = (train.log_duration - logmean)/logstd

## and maybe plotting the graph

plt.hist(train['log_duration'].values, bins = 100)
plt.show()

def plotnyc(dataset):
    xlim = (-74.03, -73.75)
    ylim = (40.63, 40.85)
    plt.scatter(dataset['pickup_longitude'].values, dataset['pickup_latitude'].values,  s = 1, alpha = 0.1)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()

##functions for calculating the distance

def haversine(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    radius = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * radius * np.arcsin(np.sqrt(d))
    return h

def L1_distance(lat1, lng1, lat2, lng2):
    a = haversine(lat1, lng1, lat1, lng2)
    b = haversine(lat1, lng1, lat2, lng1)
    return a + b




