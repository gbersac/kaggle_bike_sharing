from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.cross_validation import cross_val_predict
import math
import numpy as np
import pandas as pd
import time
import datetime
from sklearn import linear_model
import matplotlib.pyplot as plt

FILE_NAME_TRAIN = 'data.csv'
FILE_NAME_TEST = 'test.csv'

################################################################################
# Formatting data                                                              #
################################################################################

def formatData(data):
	### Refactor the `datetime` column
	def splitDateTime(s):
		date = datetime.datetime.strptime(s[0:10], "%Y-%m-%d")
		dayOfYear  = date.timetuple().tm_yday
		hour = int(s[10:13])
		weekDay = date.weekday()
		return (hour, dayOfYear, weekDay)
	data['hour'], data['dayOfYear'], data['weekDay'] = zip(*data["datetime"].map(splitDateTime))
	data.drop('datetime', axis=1, inplace=True)

	### Refactor `weekDay` to non-multi-categorial columns
	def refactorWeekDay(x):
		""" 1 = printemps , 2 = été, 3 = automne, 4 = hiver """
		if x == 0:
			return (1, 0, 0, 0, 0, 0, 0)
		if x == 1:
			return (0, 1, 0, 0, 0, 0, 0)
		if x == 2:
			return (0, 0, 1, 0, 0, 0, 0)
		if x == 3:
			return (0, 0, 0, 1, 0, 0, 0)
		if x == 4:
			return (0, 0, 0, 0, 1, 0, 0)
		if x == 5:
			return (0, 0, 0, 0, 0, 1, 0)
		if x == 6:
			return (0, 0, 0, 0, 0, 0, 1)
		print('Error', x)
	data['isMon'], data['isTue'], data['isWed'], data['isThu'], data['isFri'], data['isSat'], data['isSun'] = \
			zip(*data['weekDay'].map(refactorWeekDay))
	data.drop('weekDay', axis=1, inplace=True)

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

	### Refactor `hour` to non-multi-categorial
	def refactorHour(x):
		toReturn = [0, 0, 0, 0, 0, 0, 0, 0]
		toReturn[int(x / 3)] = 1
		return tuple(toReturn)
	data['0-2'], data['3-5'], data['6-8'], data['9-11'], data['12-14'], data['15-17'], data['18-20'], data['21-23'] \
			= zip(*data['hour'].map(refactorHour))

	return data

trainDS = formatData(pd.read_csv(FILE_NAME_TRAIN)) # DS for Data Set

# This is the files which contain the predicting value for kaggle competition
outputX = formatData(pd.read_csv(FILE_NAME_TEST)).as_matrix()

### Refactor trainDS
# Delete redundant values
# `casual` + `registered` == `count`. They are not the final value.
trainDS.drop('casual', axis=1, inplace=True)
trainDS.drop('registered', axis=1, inplace=True)

# Reorder columns
cols = [col for col in trainDS if col != 'count'] + ['count']
trainDS = trainDS[cols]

# Split dataset
# for the features
cols = [col for col in trainDS if col != 'count']
trainX = trainDS.loc[:trainDS.shape[0] * 4 / 5, cols].as_matrix()
testX = trainDS.loc[trainDS.shape[0] * 4 / 5:, cols].as_matrix()

# for the result
trainY = trainDS.loc[:trainDS.shape[0] * 4 / 5, ['count']].as_matrix()
testY = trainDS.loc[trainDS.shape[0] * 4 / 5 :, ['count']].as_matrix()

################################################################################
# Linear regression model                                                      #
################################################################################

### Compute linear regression
regr = linear_model.LinearRegression()
regr.fit(trainX, trainY)
predicted = regr.predict(testX)

### Evaluate accuracy of the computed model
# The mean square error
print("### For linear regression")
print("Residual sum of squares: %.2f"
      % np.mean((predicted - testY) ** 2))
# Variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(testX, testY))
print("")

### Plot result
fig, ax = plt.subplots()
ax.scatter(testY, predicted)
ax.plot([testY.min(), testY.max()], [testY.min(), testY.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
fig.savefig('img/final_lr.png')

################################################################################
# Random Forest model                                                          #
################################################################################

### Compute model
clf = ExtraTreesRegressor(n_estimators=10)
clf = clf.fit(trainX, np.ravel(trainY))
predicted = clf.predict(testX)

### Evaluate accuracy of the computed model
# The mean square error
print("### For random forest")
print("Residual sum of squares: %.2f" % np.mean((predicted - testY) ** 2))
# Variance score: 1 is perfect prediction
print('Variance score: %.2f' % clf.score(testX, testY))
print("")

### Plot result
fig, ax = plt.subplots()
ax.scatter(testY, predicted)
ax.plot([testY.min(), testY.max()], [testY.min(), testY.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
fig.savefig('img/final_clf.png')

################################################################################
# Output for kaggle competition                                                #
################################################################################

### Output for the kaggle submission
df = pd.DataFrame({})
df['datetime'] = pd.read_csv(FILE_NAME_TEST)['datetime']
df['count'] = pd.Series(predicted)
df.to_csv('output.csv', index=False)
