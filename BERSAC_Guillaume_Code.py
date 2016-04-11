import numpy as np
import pandas as pd
import time
import datetime

FILE_NAME = 'data.csv'

################################################################################
# Formatting data                                                              #
################################################################################

### Read data from file
data = pd.read_csv(FILE_NAME)

### Refactor the `datetime` column
def splitDateTime(s):
	date = time.mktime(datetime.datetime.strptime(s[0:10], "%Y-%m-%d").timetuple())
	dayOfYear  = datetime.datetime.strptime(s[0:10], "%Y-%m-%d").timetuple().tm_yday
	hour = int(s[10:13])
	return (date, hour, dayOfYear)

data['date'], data['hour'], data['dayOfYear'] = zip(*data["datetime"].map(splitDateTime))
data.drop('datetime', 1)

### Refactor `season` to non-multi-categorial columns

### Refactor `weather` to non-multi-categorial columns

### Reorder columns
cols = [col for col in data if col != 'count'] + ['count']
data = data[cols]

################################################################################
# Computing model                                                              #
################################################################################

### Split dataset

### Compute linear regression

### Test for overfitting

### Evaluate margin of error

################################################################################
# Prediction console                                                           #
################################################################################
