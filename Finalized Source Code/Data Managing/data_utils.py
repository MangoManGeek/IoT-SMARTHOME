import numpy as np
import math
#takes the root directory for combined.csv as an argument, returns a tuple of corresponding values for each data entry

def get_np_arr(root):
    np_arr = np.genfromtxt(root+'combined.csv', delimiter=',')
    return np_arr

def extract_data_from_csv(root):
    np_arr = get_np_arr(root)
    all_data = list()
    for time,AT,OT,RH,Bar,Light in np_arr:
        if valid_data(AT) and valid_data(OT) and valid_data(RH) and valid_data(Bar) and valid_data(Light):
            all_data.append((AT,OT,RH,Bar,Light))
    #all_data = all_data[::30]
    return all_data

#returns true if the data is a valid number
def valid_data(item):
    return item is not None and not math.isnan(item)

#gets the date for a specific index in the data
def getDate(index,root):
    i = -1
    for line in open(root+'combined.csv'):
        if(i == index):
            return line.split(",")[0]
        i += 1

#returns a dictionary that allows you to find data for a specific date
def generateDateDictionary(root):
    data = list()
    for line in open(root+'combined.csv'):
        data.append(line.split(","))
    dictionary = dict()
    for sub_list in data:
        dictionary[sub_list[0]] = sub_list[1:]
    return dictionary
