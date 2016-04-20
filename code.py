# use with python3 `python3 code.py`

from sklearn import cluster, datasets
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

# Number of cluster to create for splitting the `hour` feature.
NB_HOUR_CLUSTER = 12

# Set to true if you want to train the model with all train data and generate
# the solution to submit on kaggle
OUTPUT_KAGGLE = False

################################################################################
# Model evaluation functions                                                   #
################################################################################

def rSquared(predicted, y):
	"""
	Compute the coefficient of determination (R squared)
	Returned value between 0 and 1. The closer to 1, the better the model.
	https://en.wikipedia.org/wiki/Coefficient_of_determination
	"""
	predicted = np.ravel(predicted)
	y = np.ravel(y)
	yMean = np.mean(y)
	ttlSumSquared = np.sum((y - yMean) ** 2)
	residualSumSquared = np.sum((y - predicted) ** 2)
	return 1 - (residualSumSquared / ttlSumSquared)

def rmse(predicted, y):
	"""
	Root Mean Squared Error
	The lower it is, better is the model.
	https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError
	"""
	predicted = np.ravel(predicted)
	y = np.ravel(y)
	# inner = np.subtract(np.log(predicted + 1), np.log(y + 1)) ** 2
	inner = (predicted - y) ** 2
	return np.sqrt(np.mean(inner))

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
		year = date.year
		return (hour, dayOfYear, weekDay, year)
	data['hour'], data['dayOfYear'], data['weekDay'], data['year'] = zip(*data["datetime"].map(splitDateTime))
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

	return data

### Read from files
rawTrainData = pd.read_csv(FILE_NAME_TRAIN)
# This is the files which contain the data from which to predict for kaggle competition
rawTestData = pd.read_csv(FILE_NAME_TEST)

trainDS = formatData(rawTrainData) # DS for Data Set
outputDS = formatData(rawTestData)

### Cluster hours
KMean = cluster.KMeans(n_clusters=NB_HOUR_CLUSTER)
KMean.fit(trainDS['hour'].reshape(-1, 1), trainDS['count'].reshape(-1, 1))

def refatorHour(data):
	"""
	Transform `hour` to multiple series. One serie for each NB_HOUR_CLUSTER.
	For one row, the value of the serie which number == kmean cluster
	assignation equal one, 0 for others.
	"""
	global KMean
	floatHours = data['hour'].astype('float64')
	for i in range(0, NB_HOUR_CLUSTER):
		serieName = 'hourCluster' + str(i)
		serie = KMean.predict(floatHours.reshape(-1, 1))
		serie = pd.Series(map(lambda x: int(x == i), serie))
		data[serieName] = serie

refatorHour(trainDS)
refatorHour(outputDS)

### Refactor trainDS
# Reorder columns
cols = [col for col in trainDS if col != 'casual' and col != 'registered'] + ['casual', 'registered']
trainDS = trainDS[cols]

# Split dataset
if OUTPUT_KAGGLE:
	# for the features
	cols = [col for col in trainDS if col != 'registered' and col != 'casual' and col != 'count']
	trainX = trainDS.loc[:, cols].as_matrix()
	testX = outputDS.loc[:, cols].as_matrix()

	# for the result
	trainYCasual = trainDS.loc[:, ['casual']].as_matrix()
	trainYRegistered = trainDS.loc[:, ['registered']].as_matrix()
	trainY = trainDS.loc[:, ['count']].as_matrix()

else:
	# for the features
	cols = [col for col in trainDS if col != 'registered' and col != 'casual' and col != 'count']
	trainX = trainDS.loc[:trainDS.shape[0] * 4 / 5, cols].as_matrix()
	testX = trainDS.loc[trainDS.shape[0] * 4 / 5:, cols].as_matrix()

	# for the result
	trainYCasual = trainDS.loc[:trainDS.shape[0] * 4 / 5, ['casual']].as_matrix()
	trainYRegistered = trainDS.loc[:trainDS.shape[0] * 4 / 5, ['registered']].as_matrix()
	trainY = trainDS.loc[:trainDS.shape[0] * 4 / 5, ['count']].as_matrix()

	testYCasual = trainDS.loc[trainDS.shape[0] * 4 / 5 :, ['casual']].as_matrix()
	testYRegistered = trainDS.loc[trainDS.shape[0] * 4 / 5 :, ['registered']].as_matrix()
	testY = trainDS.loc[trainDS.shape[0] * 4 / 5 :, ['count']].as_matrix()

################################################################################
# Linear regression model                                                      #
################################################################################

### Compute linear regression
regrCasual = linear_model.LinearRegression()
regrRegistered = linear_model.LinearRegression()
regrCasual.fit(trainX, trainYCasual)
regrRegistered.fit(trainX, trainYRegistered)
predicted = regrCasual.predict(testX) +regrRegistered.predict(testX)

if not OUTPUT_KAGGLE:
	### Evaluate accuracy of the computed model
	# The mean square error
	print("### For linear regression")
	print('r squared: %.2f' % rSquared(predicted, testY))
	print('rmse: %.2f' % rmse(predicted, testY))
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
clfCasual = ExtraTreesRegressor(n_estimators=10)
clfCasual = clfCasual.fit(trainX, np.ravel(trainYCasual))
clfRegistered = ExtraTreesRegressor(n_estimators=10)
clfRegistered = clfRegistered.fit(trainX, np.ravel(trainYRegistered))
predicted = clfCasual.predict(testX) + clfRegistered.predict(testX)

if not OUTPUT_KAGGLE:
	### Evaluate accuracy of the computed model
	# The mean square error
	print("### For random forest")
	print('r squared: %.2f' % rSquared(predicted, testY))
	print('rmse: %.2f' % rmse(predicted, testY))
	print("")

	### Plot result
	fig, ax = plt.subplots()
	ax.scatter(testY, predicted)
	ax.plot([testY.min(), testY.max()], [testY.min(), testY.max()], 'k--', lw=4)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	fig.savefig('img/final_clf.png')

	# ### Plot train
	# fig, ax = plt.subplots()
	# ax.scatter(trainY, clf.predict(trainX))
	# ax.plot([trainY.min(), trainY.max()], [trainY.min(), trainY.max()], 'k--', lw=4)
	# ax.set_xlabel('Measured')
	# ax.set_ylabel('Predicted')
	# fig.savefig('img/train_clf.png')

################################################################################
# Output for kaggle competition                                                #
################################################################################

if OUTPUT_KAGGLE:
	### Output for the kaggle submission
	df = pd.DataFrame({})
	df['datetime'] = pd.read_csv(FILE_NAME_TEST)['datetime']
	df['count'] = pd.Series(predicted)
	df.to_csv('output.csv', index=False)
