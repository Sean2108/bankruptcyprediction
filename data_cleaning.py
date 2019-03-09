"""
This script will convert the original dataset (horizontally concatenated) into a list of dictionaries.
- Each element of the list represents information from 1 company
- Each company's information is stored in a dictionary

Example:
{
    Simfin ID: 247341,
    company: 1 800 FLOWERS COM INC,
    ...
    Share Price: [],
    Common Shares Outstanding: [],
    ...
    Net Change in Cash: []
}
"""

import csv
import json
import math


DATASET_DIRECTORY = './data/'
DATA_INTERVAL = 37
master_list = [] # There are 2167 companies in the dataset 
metric_list = []

with open(DATASET_DIRECTORY + 'output-comma-wide.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:     # skip first line 
            line_count += 1
            continue
        else:                   # not first line
            column_count = 0
            for value in row:
                if column_count == 0:
                    if line_count < 6: # before the dates
                        dictionary_key = value
                        column_count += 1
                    else:
                        # first col and data row = pass
                        column_count += 1
                        pass
                    # continue
                else:           # not first column
                    if line_count < 6:  # values from top rows, a lot of repeated info - just get the first time and skip?
                        if (column_count - 1) % DATA_INTERVAL == 0: 
                            if line_count == 1:
                                master_list.append({})
                            master_list[math.floor(column_count/DATA_INTERVAL)][dictionary_key] = value 
                            column_count += 1
                        else:
                            column_count += 1
                            pass
                    else:       # indicator info, each col is a financial metric. 
                        # Col quotient tells us which company it belongs to. 
                        # Col remainder tells us which metric we are looking at
                        # Each row is a date. Very likely to be empty, but we'll just append a ''
                        if line_count == 6:
                            # Create the dictionary key/value pair here
                            master_list[math.floor((column_count - 1)/DATA_INTERVAL)][value] = []
                            if (column_count - 1) < DATA_INTERVAL:
                                metric_list.append(value)
                            column_count += 1
                        else: # line_count > 7, values come in 
                            master_list[math.floor((column_count - 1)/DATA_INTERVAL)][metric_list[(column_count - 1) % DATA_INTERVAL]].append(value)
                            column_count += 1
            line_count += 1
            print(line_count)
    print(f'Processed {line_count} lines.')


print(metric_list)
print(len(metric_list))


# with open('financial_statements.json', 'w') as f:
#     json.dump(master_list, f)


import pickle
with open('financial_statements.pickle', 'wb') as f:
    pickle.dump(master_list, f)