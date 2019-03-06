# BC3409 Project: Bankruptcy Prediction

## Running the code

Remember to add the datasets into the ./data/ folder. Or change the directory written in the code. 

## Problem Statement

Using AI and deep learing on big data to perform financial evaluation of firms, with a focus on Bankruptcy Prediction.

With increasing instability and unpredictability in the financial markets, companies can see their value rise to unpredecented heights on a good day and plummet rapidly in bad ones. Not many firms will be able to survive such turbulent changes and many companies have found themselves getting into cash flow problems. For instance, many firms in the retail industry had to file for bankruptcy in 2018. With more turbulent times expected due to factors like the disruptions brought about by AI, geopolitical tensions, an ailing Chinese economy, Brexit and US-China trade wars, bankruptcy prediction will become increasing important.  

## Dataset 

Our main dataset is obtained from [SimFin](https://simfin.com/)
- 2165 companies 
    - List of company names can be found in `listofcompanies.txt'
        - `Mylan N.V.` is duplicated
    - Out of these, 46 companies are bankrupt / had prior history of bankruptcy, either in its subsidaries or the company as a whole
- 10 years of data ranging from 1/1/09 to 11/1/19


## Approach

1. Find the best financial ratios to use as features for a LSTM model to predict bankruptcy 
    - If LSTM has too many parameters, can consider GRU 
    - Do more feature engineering (MA?)
        - Negative equity
        - ROE < -50%

2. Use GAN to generate more data? 

## Side notes

Models must be more than what we used in 2406/7. e.g. GAN, LSTM

Prof suggested to look at `Net Cash Flow` (but not operating cash flow) 

[Polish dataset](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data)


### Bankruptcy


#### General

https://en.wikipedia.org/wiki/Bankruptcy_in_the_United_States

@US: Chapter 7, 11

Chapter 11 is a form of bankruptcy that involves a reorganization of a debtor's business affairs, debts, and assets. Named after the U.S. bankruptcy code 11, corporations generally file Chapter 11 if they require time to restructure their debts. This version of bankruptcy gives the debtor a fresh start.


#### Altman Z-score

Z score papers and links 
- http://pages.stern.nyu.edu/~ealtman/Zscores.pdf
- https://www.investopedia.com/terms/a/altman.asp
- https://strategiccfo.com/z-score-model/
- https://www.financewalk.com/altman-zscore/#is-this-ratio-applicable-to-all-companies


#### Ideas for expansion

To find more companies that are bankrupt, can consider
- Searching for Insolvent ones as well
- Other links:
    - http://www.bankruptcydata.com/p/bankruptcy-data-free-trial
    - https://www.macroaxis.com/invest/ratio/HDSN--Probability-Of-Bankruptcy
    - https://clark.com/shopping-retail/retailers-on-bankruptcy-watch-for-2018/
    - https://www.retaildive.com/news/10-retailers-at-risk-of-bankruptcy-q2-update/528937/
        - CreditRiskMonitor estimates the risk of a company filing for bankruptcy within 12 months based on several streams of data, including financial ratios, bond ratings, a commonly used credit analysis model (the "Merton" model) and aggregated data patterns from its own subscribers, which include credit professionals and major corporations.
        - CreditRiskMonitor uses all that data to assign a proprietary rating, called a "FRISK" score, that weighs the probability of bankruptcy. A FRISK score of one indicates a 9.99% to 50% chance of bankruptcy within 12 months, and a score of two corresponds with a 4% to 9.99% chance of bankruptcy. (The scores continue to 10, which indicates risk near zero.)
        - The following was culled from lists of retailers with FRISK scores of one or two in the second quarter 2018.
    - https://www.cbinsights.com/research/retail-apocalypse-timeline-infographic/
- Unvisited links:
    - https://en.wikipedia.org/wiki/Category:Defunct_companies_of_the_United_States


- Visited links (done!):
    - https://en.wikipedia.org/wiki/Category:Bankrupt_companies_of_the_United_States
    - https://en.wikipedia.org/wiki/Category:Companies_that_have_filed_for_bankruptcy_in_the_United_States
    - https://en.wikipedia.org/wiki/Category:Companies_that_have_filed_for_Chapter_11_bankruptcy_by_year
        - Done for 2006-2019
    - https://en.wikipedia.org/wiki/Category:Companies_that_have_filed_for_Chapter_7_bankruptcy 
    - https://www.ranker.com/list/list-of-bankrupt-companies-and-subsidiaries/business-and-company-info
    - https://www.infoplease.com/business-finance/business/largest-us-bankruptcies-1980-present
    - https://www.businessinsider.sg/retail-bankruptcies-list-this-year-2018-4/?r=US&IR=T
    - https://www.businessinsider.com/bankruptcies-expected-this-year-2018-4/?IR=T/
    - https://www.finance101.com/bankruptcy/
