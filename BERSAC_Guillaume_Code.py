import math
import numpy as np
import pandas as pd
import time
import datetime
from sklearn import linear_model

FILE_NAME = 'data.csv'

################################################################################
# Formatting data                                                              #
################################################################################

### Read data from file
data = pd.read_csv(FILE_NAME)

### Refactor the `datetime` column
def splitDateTime(s):
	dayOfYear  = datetime.datetime.strptime(s[0:10], "%Y-%m-%d").timetuple().tm_yday
	hour = int(s[10:13])
	return (hour, dayOfYear)
data['hour'], data['dayOfYear'] = zip(*data["datetime"].map(splitDateTime))
data.drop('datetime', axis=1, inplace=True)

### Refactor `season` to non-multi-categorial columns
def refactorSeason(x):
	""" 1 = printemps , 2 = été, 3 = automne, 4 = hiver """
	if x == 1:
		return (1, 0, 0, 0)
	if x == 2:
		return (0, 1, 0, 0)
	if x == 3:
		return (0, 0, 1, 0)
	if x == 4:
		return (0, 0, 0, 1)
	return (0, 0, 0, 0)
data['isSpring'], data['isSummer'], data['isAutumn'], data['isWinter'] = \
		zip(*data['season'].map(refactorSeason))
data.drop('season', axis=1, inplace=True)

### Refactor `weather` to non-multi-categorial columns
def refactorWeather(x):
	""" 1: Dégagé, 2 : Brouillard, 3 : Légère pluie, 4 : Fortes averses """
	if x == 1:
		return (1, 0, 0, 0)
	if x == 2:
		return (0, 1, 0, 0)
	if x == 3:
		return (0, 0, 1, 0)
	if x == 4:
		return (0, 0, 0, 1)
data['brightWeather'], data['fogWeather'], data['rainWeather'], data['heavyRainWeather'] = \
		zip(*data['weather'].map(refactorWeather))
data.drop('weather', axis=1, inplace=True)

### Delete redundant values
# `casual` + `registered` == `count`. They are not the final value.
data.drop('casual', axis=1, inplace=True)
data.drop('registered', axis=1, inplace=True)

### Reorder columns
cols = [col for col in data if col != 'count'] + ['count']
# data['count'], _ = zip(*data['count'].map(lambda x: (math.sqrt(x), 1)))
data = data[cols]

################################################################################
# Computing model                                                              #
################################################################################

### Split dataset
# for the features
cols = [col for col in data if col != 'count']
trainX = data.loc[:data.shape[0] * 2 / 3, cols].as_matrix()
testX = data.loc[data.shape[0] * 2 / 3:, cols].as_matrix()

# for the result
trainY = data.loc[:data.shape[0] * 2 / 3, ['count']].as_matrix()
testY = data.loc[data.shape[0] * 2 / 3 :, ['count']].as_matrix()

### Compute linear regression
regr = linear_model.LinearRegression()
regr.fit(trainX, trainY)

### Evaluate accuracy of the computed model
for i in range(0, len(regr.coef_[0])):
	print(data.columns.values[i], '->', regr.coef_[0][i])

# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(testX) - testY) ** 2))
# Variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(testX, testY))
