#import packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

#Loading the Dataset

dataset_train = pd.read_csv('elec.csv') # dataset: http://jmcauley.ucsd.edu/data/amazon/            https://nijianmo.github.io/amazon/

# this dataset contains NePS and salesrank information

training_set = dataset_train.iloc[:, 1:2].values
#dataset_train.head()

#Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#Creating Data with Timesteps
X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Building the LSTM

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_absolute_error')

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#Predicting Future Stock using the Test Set

dataset_test = pd.read_csv('electest.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['x1'], dataset_test['x1']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_salesrank = regressor.predict(X_test)
predicted_salesrank = sc.inverse_transform(predicted_stock_price)

#Plotting 
plt.plot(real_salesrank, color = 'black', label = 'SalesRank')
plt.plot(predicted_salesrank, color = 'green', label = 'Predicted SalesRank')
plt.title('SalesRank Prediction')
plt.xlabel('NePS')
plt.ylabel('SalesRank')
plt.legend()
plt.show()
