# BC3409 Project: Bankruptcy Prediction

## Running the code

Remember to add the datasets into the ./data/ folder. Or change the directory written in the code.
Please run `simfinmodel.ipynb` to run our LSTM/GRU models to predict bankruptcy on our dataset.

## Problem Statement

Using AI and deep learning to perform financial evaluation of firms (with a focus on Bankruptcy Prediction) via big data.

## Dataset 

Our main dataset is obtained from [SimFin](https://simfin.com/)
- 2165 companies 
    - List of company names can be found in `listofcompanies.txt'
        - `Mylan N.V.` is duplicated
    - Out of these, 49 companies are bankrupt / had prior history of bankruptcy, either in its subsidaries or the company as a whole
- 10 years of data ranging from 1/1/09 to 11/1/19
- 42 attributes of financial data (including financial information of company and market ratios)

Our secondary dataset is obtained from [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data)
- 5190 companies
    - 1 year of data ranging from 1/1/12 to 12/1/12
    - 64 attributes of financial data (different from our main dataset)

Our final dataset used after preprocessing based on our main dataset is `data_with_ratios.pickle`.





# Misc notes

changed to sgd
added early stopping 

should mention keras



try to use single thread
then, lower learning rate? val_loss spazing too much 


(base) C:\Users\manza\Documents\GitHub\bankruptcyprediction>python alt_simfinmodel.py
WARNING:tensorflow:From alt_simfinmodel.py:13: The name tf.set_random_seed is deprecated. Please use tf.compat.v1.set_random_seed instead.

Using TensorFlow backend.
WARNING:tensorflow:From alt_simfinmodel.py:29: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.

['bankrupt', 'company', 'debt_equity', 'equity', 'solvency', 'ticker', 'x10', 'x15', 'x16', 'x17', 'x2', 'x29', 'x3', 'x41', 'x50', 'x55', 'x7', 'x8']
WARNING:tensorflow:From alt_simfinmodel.py:210: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

WARNING:tensorflow:From alt_simfinmodel.py:210: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

OMP: Info #214: KMP_AFFINITY: OS proc to physical thread map:
OMP: Info #171: KMP_AFFINITY: OS proc 0 maps to package 0 core 0
OMP: Info #171: KMP_AFFINITY: OS proc 1 maps to package 0 core 1
OMP: Info #171: KMP_AFFINITY: OS proc 2 maps to package 0 core 2
OMP: Info #171: KMP_AFFINITY: OS proc 3 maps to package 0 core 3
OMP: Info #250: KMP_AFFINITY: pid 7180 tid 8600 thread 0 bound to OS proc set 0
OMP: Info #250: KMP_AFFINITY: pid 7180 tid 3464 thread 1 bound to OS proc set 1
OMP: Info #250: KMP_AFFINITY: pid 7180 tid 4260 thread 2 bound to OS proc set 2
OMP: Info #250: KMP_AFFINITY: pid 7180 tid 10356 thread 3 bound to OS proc set 3
WARNING:tensorflow:From C:\Users\manza\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:From C:\Users\manza\Anaconda3\lib\site-packages\keras\optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

WARNING:tensorflow:From C:\Users\manza\Anaconda3\lib\site-packages\tensorflow\python\ops\nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in 
a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 1157 samples, validate on 205 samples