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
