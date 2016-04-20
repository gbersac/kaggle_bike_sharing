# use with python3 `python3 eda.py`
"""
A python script to crate plots to have an overview of the data in data.csv.
eda for Exploratory Data Analisys
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import time
import datetime

FILE_NAME = 'data.csv'

# rearange data
def splitDateTime(s):
	date = datetime.datetime.strptime(s[0:10], "%Y-%m-%d")
	dayOfYear  = date.timetuple().tm_yday
	hour = int(s[10:13])
	weekDay = date.weekday()
	year = date.year
	return (hour, dayOfYear, weekDay, year)
data = pd.read_csv(FILE_NAME)
data['hour'], data['dayOfYear'], data['weekDay'], data['year'] = zip(*data["datetime"].map(splitDateTime))
data.drop('datetime', 1)

# delete redundant columns
data.drop('casual', axis=1, inplace=True)
data.drop('registered', axis=1, inplace=True)


# reorder columns
cols = [col for col in data if col != 'count'] + ['count']
data = data[cols]

# scatter matrix plot
print('end loading data')
plot = scatter_matrix(data, alpha=0.2, figsize=(14, 14), diagonal='kde')
plt.savefig('img/scatter_matrix.png')

# count histogram
ax = data['count'].hist()
fig = ax.get_figure()
fig.savefig('img/count_histo.png')
