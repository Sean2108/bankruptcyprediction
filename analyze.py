import pickle

pickle_in = open("./data/financial_statements.pickle","rb")
finstat = pickle.load(pickle_in)
print(len(finstat))

finstat[0].keys() # see all variables