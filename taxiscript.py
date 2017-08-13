# this is a script for the kaggle competition

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from datetime import timedelta
import datetime as dt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')



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

##converting everything to datetime

def  cleandata(dataset):
    dataset['pickup_datetime'] = pd.to_datetime(dataset.pickup_datetime)
    dataset.loc[:, 'pickup_date'] = dataset['pickup_datetime'].dt.date
    #binary files to boolean
    dataset['store_and_fwd_flag'] = 1 * (dataset.store_and_fwd_flag.values == 'Y')
    dataset.loc[:, 'haversine'] = haversine(dataset['pickup_latitude'].values, dataset['pickup_longitude'].values, dataset['dropoff_latitude'].values, dataset['dropoff_longitude'].values)
    dataset.loc[:, 'L1_distance'] = L1_distance(dataset['pickup_latitude'].values, dataset['pickup_longitude'].values, dataset['dropoff_latitude'].values, dataset['dropoff_longitude'].values)

    #getting weekday values
    dataset.loc[:, 'weekday'] = dataset.pickup_date.apply(lambda x: x.weekday())
    dataset.loc[:, 'hour'] = dataset.pickup_datetime.apply(lambda x: x.hour)

    dataset = dataset.drop('vendor_id', axis = 1)
    dataset = dataset.drop('id', axis = 1)
    dataset = dataset.drop('pickup_date', axis = 1)
    dataset = dataset.drop('pickup_datetime', axis = 1)
    return dataset




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


   # newtrain = newtrain.drop('dropoff_datetime', axis = 1)

train = cleandata(train)
test = cleandata(test)

#normalizing the L1 distance

l1train = train['L1_distance'].apply(lambda x: np.log(x + 0.01))

l1train.hist(bins = 50)




