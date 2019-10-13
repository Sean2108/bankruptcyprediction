## Modules
import os
os.environ['PYTHONHASHSEED'] = '700'

import random as rn
rn.seed(700)

import numpy as np
from numpy.random import seed
seed(700)

import tensorflow as tf
from tensorflow import set_random_seed
set_random_seed(700)
# tf.compat.v1.set_random_seed(700)

import pickle
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

# %matplotlib inline

#set up matplotlib
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (30, 15),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)


## Constants 

MIN_ROWS = 5
ZSCORE = 1.8

EPOCH = 100
BATCH_SIZE = 8
LSTM_UNITS = 60
DROPOUT = 0.2
NUM_LSTM_LAYERS = 4
LEARNING_RATE = 0.0005 # 0.001

PATIENCE = 10

MODEL_TO_RUN = 'gru' # must be in small letters 

# lstm, gru
# 42 - 60, 100
# 100 - 73, 67
# 200 - 74, 67
# 300 - 70, 16
# 400 - 73, 75 -- 65, 87
# 500 - 69, 12 -- 72, 75
# 600 - 75, 54 -- 76, 37
# 700 - 59, 100

## Functions
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def train_model(x_train, y_train, units, dropout, num_lstm_layers, model_type):
    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])) if model_type == 'lstm' 
            else GRU(units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])) if model_type == 'gru' 
            else SimpleRNN(units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
            Dropout(dropout)
        ] +
        [
            LSTM(units, return_sequences=True) if model_type == 'lstm' else 
            GRU(units, return_sequences=True) if model_type == 'gru' else
            SimpleRNN(units, return_sequences=True),
            Dropout(dropout)
        ] * (num_lstm_layers - 2) +
        [
            LSTM(units) if model_type == 'lstm' else 
            GRU(units) if model_type == 'gru' else
            SimpleRNN(units),
            Dropout(dropout),
            Dense(1, activation = 'sigmoid')
        ])



    return model#, history

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
    
with open('data_with_ratios.pickle', 'rb') as fp:
    data = pickle.load(fp)

new_data = []
bankrupt = []

cols = data[0].keys()

for obj in data:
    if (len(obj.keys()) <= 28):
        continue
    cols = intersection(cols, obj.keys())
cols = sorted(cols)
print(cols)

ground_truth = []
ground_bankrupt = []

# Build a list of companies that are really bankrupt
# 'data' already contains labels of truly bankrupt companies

for obj in data:
    if not obj['bankrupt']:
        continue
    valid = True
    new_obj = []
    for i in range(MIN_ROWS):
        new_vals = []
        for key in cols:
            if key not in obj:
                valid = False
                break
            if (isinstance(obj[key], list) and key not in ['equity', 'roe', 'bad_solvency', 'debt_equity']):
                if len(obj[key]) < MIN_ROWS:
                    valid = False
                    break
                new_vals.append(obj[key][i]['value'])
        new_obj.append(new_vals)
    if (valid):
        ground_bankrupt.append(True)
        ground_truth.append(new_obj)
  
for obj in data:
    if obj['bankrupt']:
        continue
    valid = True
    new_obj = []
    for i in range(MIN_ROWS):
        new_vals = []
        for key in cols:
            if key not in obj:
                valid = False
                break
            if (isinstance(obj[key], list) and key not in ['equity', 'roe', 'bad_solvency', 'debt_equity']):
                if len(obj[key]) < MIN_ROWS:
                    valid = False
                    break
                new_vals.append(obj[key][i]['value'])
        new_obj.append(new_vals)
    if (valid):
#         ban = obj['bankrupt']
#         if 'zscore' in obj:
#             for zscore in obj['zscore']:
#                 if zscore['value'] < ZSCORE:
#                     ban = True
        neg_equity = False
        neg_roe = False
        bad_solvency = False
        bad_debt = False
        if 'equity' in obj:
            for equity in obj['equity']:
                if equity['value'] < 0:
                    neg_equity = True
        if 'roe' in obj:
            current_roe = None
            for roe in obj['roe']:
                if current_roe and roe['value'] < current_roe / 2:
                    neg_roe = True
                current_roe = roe['value']
        if 'bad_solvency' in obj:
            for solv in obj['bad_solvency']:
                if solv['value'] < 2:
                    bad_solvency = True
        if 'debt_equity' in obj:
            for de in obj['debt_equity']:
                if de['value'] > 2:
                    bad_debt = True
        bankrupt.append((neg_equity or neg_roe) and (bad_debt or bad_solvency))
        new_data.append(new_obj)

# Data Prep

continue_flag = True

while continue_flag:

    results_test = []
    results_ground_truth = []

    for ix in range(10):

        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

        arr = np.array(new_data)
        test_arr = np.array(ground_truth)
        print("Dataset Dimensions" + str(arr.shape))

        scalers = {}
        for i in range(arr.shape[2]):
            scalers[i] = MinMaxScaler(feature_range=(0,1))
            arr[:, :, i] = scalers[i].fit_transform(arr[:, :, i])
            test_arr[:, :, i] = scalers[i].transform(test_arr[:, :, i])

        x_train, x_test, y_train, y_test = train_test_split(arr, bankrupt, test_size=0.15, random_state=42)

        print('bankrupt:' + str(len([True for b in bankrupt if b])))
        print('not bankrupt:' + str(len([True for b in bankrupt if not b])))

        # Train model
        lstm_model = train_model(x_train, y_train, LSTM_UNITS, DROPOUT, NUM_LSTM_LAYERS, MODEL_TO_RUN)
        # lstm_model, lstm_history = train_model(x_train, y_train, LSTM_UNITS, DROPOUT, NUM_LSTM_LAYERS, MODEL_TO_RUN, EPOCH, BATCH_SIZE, test_arr, ground_bankrupt)
        # plot_loss(lstm_history)

        callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE)]
    
        lstm_model.compile(optimizer=optimizers.Adam(lr=LEARNING_RATE),loss='binary_crossentropy', metrics=['accuracy'])
        history = lstm_model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks, validation_data=(x_test, y_test))

        early_stopping_epoch = callbacks[0].stopped_epoch - PATIENCE + 1 # keras gives the 0-index value of the epoch, so +1. -5 because of patience?
        print('Early stopping epoch: ' + str(early_stopping_epoch))

        
        # Evaluate model and predict data on TEST 
        print("******Evaluating TEST set*********")

        scores = lstm_model.evaluate(x_test, y_test)
        print("model: \n%s: %.2f%%" % (lstm_model.metrics_names[1], scores[1]*100))
        results_test.append(scores[1]*100)

        y_predict = lstm_model.predict_classes(x_test)
        cm = confusion_matrix(y_test, y_predict)
        print(cm)

        try:
            tn, fp, fn, tp = cm.ravel()
            print(tn, fp, fn, tp)
        except ValueError:
            print("100% accuracy, no CM to print")

        fpr_BDmodel, tpr_BDmodel, thresholds_BDmodel = roc_curve(y_test, y_predict)
        auc_BDmodel = auc(fpr_BDmodel, tpr_BDmodel)
        print("AUC: " + str(auc_BDmodel))

        # from keras import backend
        K.clear_session()

        lstm_model = train_model(x_train, y_train, LSTM_UNITS, DROPOUT, NUM_LSTM_LAYERS, MODEL_TO_RUN)

        lstm_model.compile(optimizer=optimizers.Adam(lr=LEARNING_RATE),loss='binary_crossentropy', metrics=['accuracy'])
        history_second = lstm_model.fit(arr, bankrupt, epochs=early_stopping_epoch, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks, validation_data=(test_arr, ground_bankrupt))

        best_epoch = history_second.history['val_acc'].index(max(history_second.history['val_acc']))
        best_val_loss = history_second.history['val_loss'][best_epoch]
        best_val_acc = history_second.history['val_acc'][best_epoch]

        with open('results_ground_truth' + MODEL_TO_RUN + '_detailed.csv', 'a') as out_stream:
            out_stream.write(str(ix) + ', ' + str(best_epoch) + ', ' + str(best_val_loss) + ', ' + str(best_val_acc) + '\n')


        # Evaluate model and predict data on GROUND TRUTH
        print("******Evaluating GROUND TRUTH*********")
        scores = lstm_model.evaluate(test_arr, ground_bankrupt)
        print("model: \n%s: %.2f%%" % (lstm_model.metrics_names[1], scores[1]*100))
        results_ground_truth.append(scores[1]*100)



        y_predict = lstm_model.predict_classes(test_arr)
        cm = confusion_matrix(ground_bankrupt, y_predict)
        print(cm)
        
        try:
            tn, fp, fn, tp = cm.ravel()
            print(tn, fp, fn, tp)
        except ValueError:
            print("100% accuracy, no CM to print")
        
        print(str(ix) + ' done!')
        print(results_test)
        print(results_ground_truth)



    import statistics

    print(results_test)
    print("Average test accuracy: " + str(statistics.mean(results_test)))
    print("Average test stdev: " + str(statistics.stdev(results_test)))

    with open('results_test' + MODEL_TO_RUN + '.csv', 'a') as out_stream:
        out_stream.write(str(statistics.mean(results_test)) + ', ' + str(statistics.stdev(results_test)) + ', ' + str(results_test) + '\n')

    print(results_ground_truth)
    print("Average ground truth accuracy: " + str(statistics.mean(results_ground_truth)))
    print("Average ground stdev: " + str(statistics.stdev(results_ground_truth)))

    with open('results_ground_truth' + MODEL_TO_RUN + '.csv', 'a') as out_stream:
        out_stream.write(str(statistics.mean(results_ground_truth)) + ', ' + str(statistics.stdev(results_ground_truth)) + ', ' + str(results_ground_truth) + '\n')

    continue_flag = False
