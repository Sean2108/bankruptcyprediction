## Modules
import os
os.environ['PYTHONHASHSEED'] = '0'

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import tensorflow as tf
tf.set_random_seed(0)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

import pickle
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras import backend as K

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split


## Constants 

MIN_ROWS = 5
ZSCORE = 1.8

EPOCH = 200
BATCH_SIZE = 8
LSTM_UNITS = 10
DROPOUT = 0.2
NUM_LSTM_LAYERS = 2
LEARNING_RATE = 0.005 # 0.001

PATIENCE = 10
COOLDOWN = 5 

MODEL_TO_RUN = 'gru' # must be in small letters 


## Functions

class ReduceLROnPlateau_imbalanced(tf.keras.callbacks.Callback):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    # Example
    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```
    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs that produced the monitored
            quantity with no improvement after which training will
            be stopped.
            Validation quantities may not be produced for every
            epoch, if the validation frequency
            (`model.fit(validation_freq=5)`) is greater than one.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        min_delta: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """
    import warnings

    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0,
                 **kwargs):
        super(ReduceLROnPlateau_imbalanced, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau_imbalanced '
                             'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn('`epsilon` argument is deprecated and '
                          'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.monitor_op_alt = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.monitor_op_alt = lambda a, b: np.less(a, b - self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best): # if a < b - d, we are in the correct direction, reduce LR 
                # self.best = current # we only want to compare to previous?

                if not self.in_cooldown():# and self.best != np.Inf and self.best != -np.Inf:
                    self.best = current # reset reference to current score
                    self.wait += 1
                    if self.wait >= self.patience:
                        old_lr = float(K.get_value(self.model.optimizer.lr))
                        if old_lr > self.min_lr:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            K.set_value(self.model.optimizer.lr, new_lr)
                            if self.verbose > 0:
                                print('\nEpoch %05d: Correct direction. ReduceLROnPlateau_imbalanced reducing '
                                    'learning rate to %s.' % (epoch + 1, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0

            elif self.monitor_op_alt(current, self.best): # if a > b + d, it's going in the wrong direction (significant magnitude)

                if not self.in_cooldown():# and self.best != np.Inf and self.best != -np.Inf:
                    self.best = current # reset reference to current score
                    self.wait += 1
                    if self.wait >= self.patience:
                        old_lr = float(K.get_value(self.model.optimizer.lr))
                        if old_lr > self.min_lr:
                            new_lr = old_lr / self.factor
                            new_lr = max(new_lr, self.min_lr)
                            K.set_value(self.model.optimizer.lr, new_lr)
                            if self.verbose > 0:
                                print('\nEpoch %05d: Wrong direction. ReduceLROnPlateau_imbalanced increasing '
                                    'learning rate to %s.' % (epoch + 1, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0

            else: # won't get to here unless elif block is commented out 

                self.best = current
                self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


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

    return model

def prepare_data():
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


    return (new_data, bankrupt, ground_truth, ground_bankrupt)

results_test = []
results_ground_truth = []
early_stopping_epoch_list = []	

(new_data, bankrupt, ground_truth, ground_bankrupt) = prepare_data()

for ix in range(10):

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # Prepare dataset
    arr = np.array(new_data)
    test_arr = np.array(ground_truth)
    print("Dataset Dimensions" + str(arr.shape))

    scalers = {}
    for i in range(arr.shape[2]):
        scalers[i] = MinMaxScaler(feature_range=(0,1))
        arr[:, :, i] = scalers[i].fit_transform(arr[:, :, i])
        test_arr[:, :, i] = scalers[i].transform(test_arr[:, :, i])

    x_train, x_test, y_train, y_test = train_test_split(arr, bankrupt, test_size=0.15, random_state=ix)

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.lr
        return lr

    # def imbalanced_learning_rate():
    #     """
    #     Cut the learning rate if we get a significant drop in loss 
    #     """
    #     SIG_THRESHOLD = 0.03
    #     if len(history_loss) < 2:
    #         return LEARNING_RATE
    #     print(history_loss[-1])
    #     print(history_loss[-2])
    #     if history_loss[-1] - history_loss[-2] > SIG_THRESHOLD:
    #         return 


    # class LossHistory(keras.callbacks.Callback):
    #     def on_train_begin(self, logs={}):
    #         self.losses = [1,1]

    #     def on_epoch_end(self, batch, logs={}):
    #         self.losses.append(logs.get('loss'))

    # class ValLossHistory(keras.callbacks.Callback):
    #     def on_train_begin(self, logs={}):
    #         self.losses = [1,1]

    #     def on_epoch_end(self, batch, logs={}):
    #         self.losses.append(logs.get('val_loss'))

    # Train model
    lstm_model = train_model(x_train, y_train, LSTM_UNITS, DROPOUT, NUM_LSTM_LAYERS, MODEL_TO_RUN)
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE, min_lr=0, mode='min', verbose=1), ReduceLROnPlateau_imbalanced(monitor='val_loss', factor=0.5, patience=0, min_lr=1e-5, mode='min', min_delta=0.015, verbose=1)]#, LearningRateScheduler()]
    opt = optimizers.Adam(lr=LEARNING_RATE, clipnorm=1., decay=LEARNING_RATE/EPOCH)
    lr_metric = get_lr_metric(opt)
    lstm_model.compile(optimizer=opt,loss='binary_crossentropy', metrics=['accuracy', lr_metric])
    history = lstm_model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks, validation_data=(x_test, y_test))

    early_stopping_epoch = callbacks[0].stopped_epoch - PATIENCE + 1 # keras gives the 0-index value of the epoch, so +1
    print('Early stopping epoch: ' + str(early_stopping_epoch))
    early_stopping_epoch_list.append(early_stopping_epoch)
    if early_stopping_epoch < 0:
        early_stopping_epoch = 100
    
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

    K.clear_session()

    lstm_model = train_model(x_train, y_train, LSTM_UNITS, DROPOUT, NUM_LSTM_LAYERS, MODEL_TO_RUN)
    callbacks = [ReduceLROnPlateau(monitor='loss', factor=0.2, patience=PATIENCE, min_lr=0, mode='min', verbose=1), ReduceLROnPlateau_imbalanced(monitor='loss', factor=0.5, patience=0, min_lr=1e-5, mode='min', min_delta=0.015, verbose=1)]
    opt = optimizers.Adam(lr=LEARNING_RATE, clipnorm=1., decay=LEARNING_RATE/EPOCH)
    lr_metric = get_lr_metric(opt)
    lstm_model.compile(optimizer=opt,loss='binary_crossentropy', metrics=['accuracy', lr_metric])
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
    out_stream.write(str(statistics.mean(results_test)) + ', ' + str(statistics.stdev(results_test)) + ', ' + str(results_test) + ', ' + str(early_stopping_epoch_list) + '\n')

print(results_ground_truth)
print("Average ground truth accuracy: " + str(statistics.mean(results_ground_truth)))
print("Average ground stdev: " + str(statistics.stdev(results_ground_truth)))

with open('results_ground_truth' + MODEL_TO_RUN + '.csv', 'a') as out_stream:
    out_stream.write(str(statistics.mean(results_ground_truth)) + ', ' + str(statistics.stdev(results_ground_truth)) + ', ' + str(results_ground_truth) + ', ' + str(early_stopping_epoch_list) + '\n')
