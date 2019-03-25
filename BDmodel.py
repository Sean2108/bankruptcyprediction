## Modules
from tensorflow import set_random_seed

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

from scipy.io import arff

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import keras.backend as K

## Functions
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name == 'class':
            continue
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

## Constants
SEED = 10
np.random.seed(2019)
set_random_seed(2019)

data = arff.loadarff('./data/5year.arff')
df = pd.DataFrame(data[0])
# print(df.head())
df = df.dropna(axis=0, how='any')
# print(df)
df = normalize(df)

# epoch = 50
# learning_rate = 0.01
# dropout = 0.1
# class_weights = {0: 1., 1: 35.} # or use 32

epoch = 100
learning_rate = 0.008
dropout = 0.0
class_weights = {0: 1., 1: 1.}

print('learning_rate: ' + str(learning_rate), 'dropout: ' + str(dropout))

# Train/test split
x = df.values[:,0:64]
y = df.values[:,64]

y = np.asarray([int(i) for i in y])

# print(x.value_counts())
# print(y.value_counts())

# from imblearn.over_sampling import SMOTE
# smt = SMOTE()
# print(y)
# x, y = smt.fit_sample(x, y)


print(len([i for i in y if i==0]))
print(len([i for i in y if i==1]))


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Model setup
look_back = 1 
num_features = 64 

nb_samples = x_train.shape[0] - look_back
x_train_reshaped = np.zeros((nb_samples, look_back, num_features))
y_train_reshaped = np.zeros((nb_samples))

def compute_binary_specificity(y_pred, y_true):

    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    
    return specificity

def specificity_loss_wrapper():
    """
    A wrapper to create and return a function which computes the specificity loss, as (1 - specificity)
    """
    def specificity_loss(y_true, y_pred):
        return 1.0 - compute_binary_specificity(y_true, y_pred)

    return specificity_loss


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


BDmodel = Sequential()
# BDmodel.add(LSTM(64, input_shape=(look_back,num_features)))
BDmodel.add(LSTM(64, input_shape=(look_back,num_features), return_sequences = True))
BDmodel.add(Dropout(dropout, seed=SEED))
BDmodel.add(LSTM(32, return_sequences = False))
# BDmodel.add(LSTM(16))
BDmodel.add(Dense(1, activation = 'sigmoid'))

callbacks = [EarlyStopping(monitor='val_loss', patience=10)]
sgd = optimizers.SGD(lr=learning_rate, clipnorm=1.)
spec_loss = specificity_loss_wrapper()
BDmodel.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics=['accuracy'])
BDmodel.fit(x_train_reshaped, y_train_reshaped, batch_size = 32, validation_split= 0.3, epochs = epoch, class_weight= class_weights, callbacks=callbacks)

# Evaluate model and predict data
scores = BDmodel.evaluate(x_test_reshaped, y_test_reshaped)
print("BDmodel: \n%s: %.2f%%" % (BDmodel.metrics_names[1], scores[1]*100))

y_predict = BDmodel.predict_classes(x_test_reshaped)
cm = confusion_matrix(y_test_reshaped, y_predict)
print(cm)

fpr_BDmodel, tpr_BDmodel, thresholds_BDmodel = roc_curve(y_test_reshaped, y_predict)
auc_BDmodel = auc(fpr_BDmodel, tpr_BDmodel)
print(auc_BDmodel)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_BDmodel, tpr_BDmodel, label='BDmodel (area = {:.3f})'.format(auc_BDmodel))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


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
 