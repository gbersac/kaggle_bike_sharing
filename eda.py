#! /nfs/2013/g/gbersac/anaconda/bin/python3

"""
A python script to have an overview of the data in data.csv.
eda for Exploratory Data Analisys
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import time
import datetime

FILE_NAME = 'data.csv'

# rearange data
def splitDateTime(s):
	date = time.mktime(datetime.datetime.strptime(s[0:10], "%Y-%m-%d").timetuple())
	dayOfYear  = datetime.datetime.strptime(s[0:10], "%Y-%m-%d").timetuple().tm_yday
	hour = int(s[10:13])
	return (date, hour, dayOfYear)

data = pd.read_csv(FILE_NAME)
data['date'], data['hour'], data['dayOfYear'] = zip(*data["datetime"].map(splitDateTime))
data.drop('datetime', 1)

# reorder columns
cols = [col for col in data if col != 'count'] + ['count']
data = data[cols]


# save plot
print('end loading data')
plot = scatter_matrix(data, alpha=0.2, figsize=(14, 14), diagonal='kde')
plt.savefig('scatter_matrix.png')
