import pickle

def open_dataset():
    pickle_in = open("./data/financial_statements.pickle","rb")
    finstat = pickle.load(pickle_in)
    print(len(finstat))
    print(finstat[0].keys()) # see all variables

def write_date_to_pickle():
    """
    Input: dates.txt
    Output: A pickle file contains dates stored in a list, written to ./data
    """
    dates_list = []
    with open('dates.txt', 'r') as f:
        for l in f:
            dates_list.append(l.strip())

    # print(dates_list)

    with open('dates.pickle', 'wb') as f:
        pickle.dump(dates_list, f)

    return dates_list

def get_dates_from_pickle():
    """
    """
    pickle_in = open("dates.pickle","rb")
    dates_list = pickle.load(pickle_in)
    return dates_list

if __name__ == '__main__':
    write_date_to_pickle()