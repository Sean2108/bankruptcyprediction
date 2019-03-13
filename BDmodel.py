# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:02:46 2019
"""

# Create your first MLP in Keras
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from scipy.io import arff
import numpy as np
import pickle
import pandas as pd

np.random.seed(2019)
set_random_seed(2019)

data = arff.loadarff('./data/5year.arff')
df = pd.DataFrame(data[0])
df.head()
df = df.dropna(axis=0, how='any')
df

# Train/test split
x = df.values[:,0:64]
y = df.values[:,64]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Model setup
look_back = 1 # as that is how y was generated (i.e. sum last three steps)
num_features = 64 # in this case: 64 features x1, x2... x64

nb_samples = x_train.shape[0] - look_back
x_train_reshaped = np.zeros((nb_samples, look_back, num_features))
y_train_reshaped = np.zeros((nb_samples))

for i in range(nb_samples):
    y_position = i + look_back
    x_train_reshaped[i] = x_train[i:y_position]
    y_train_reshaped[i] = y_train[y_position]

nb_samples = x_test.shape[0] - look_back
x_test_reshaped = np.zeros((nb_samples, look_back, num_features))
y_test_reshaped = np.zeros((nb_samples))    

for i in range(nb_samples):
    y_position = i + look_back
    x_test_reshaped[i] = x_test[i:y_position]
    y_test_reshaped[i] = y_test[y_position]

# Create model
BDmodel = Sequential()
BDmodel.add(LSTM(128, input_shape=(look_back,num_features), return_sequences = True))
BDmodel.add(LSTM(64, input_shape=(look_back,num_features)))
BDmodel.add(Dense(1, activation = 'sigmoid'))
BDmodel.compile(optimizer = 'rmsprop', loss = 'mse', metrics=['accuracy'])
BDmodel.fit(x_train_reshaped, y_train_reshaped, batch_size = 32, validation_split= 0.3, epochs = 100)
print(BDmodel.summary())

# Test model
BDmodelpredict_train = BDmodel.predict_classes(x_train_reshaped)
BDmodelpredict_test = BDmodel.predict_classes(x_test_reshaped)

scores = BDmodel.evaluate(x_train_reshaped, y_train_reshaped)
print("BDmodel: \n%s: %.2f%%" % (BDmodel.metrics_names[1], scores[1]*100))


simfin_data = pd.read_pickle("./data_with_ratios.pickle")
simfin_df = pd.DataFrame(simfin_data[0])


'''
simfin_data = pd.read_pickle("./data_with_ratios.pickle")
simfin_data.info

x_train = np.reshape(x_train, (x_train.shape[0], look_back, num_features))
y_train = np.reshape(y_train, (x_train.shape[0], look_back, num_features))

def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back+1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back - 1, :])
    return np.array(dataX), np.array(dataY)

'''
 