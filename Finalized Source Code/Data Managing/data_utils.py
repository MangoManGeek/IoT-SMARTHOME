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


def getDataAtIndex(tuple_list,index):
    temps = list()
    for item in tuple_list:
        temps.append(item[index])
    return temps

#separates list of values into training and testing portions with a gap between them equal to 10% of the dataset
#trainingStart and testingEnd define the bounds within the dataset for data that will be used
def separateData(data,trainingSize,trainingStart,testingEnd):
    gap_between_training_testing = int(len(data) * 0.1)
    #determines of the size of the temporal gap between training and testing
    trainingStartIndex = int(len(data) * trainingStart)
    testingEndIndex = int(len(data) * testingEnd)
    interior_length = testingEndIndex - trainingStartIndex - gap_between_training_testing
    #[interior_length] is equal to the size of training and testing data combined
    num_training = int(trainingSize * interior_length)
    #sets the size of the training set to be based on the desired training-to-testing ratio and the available data to be used for analysis
    num_testing = interior_length - trainingSize
    #sets testing to whatever is left
    trainingEndIndex = trainingStartIndex + num_training
    testingStartIndex = trainingEndIndex+gap_between_training_testing
    training_data = data[trainingStartIndex:trainingEndIndex]
    testing_data = data[testingStartIndex:testingEndIndex]
    return training_data,testing_data

#calculates the cost, given a list of predictions and a list of actual values
#assumes that each list is a one-dimensional list of numbers
def calculate_cost(predictions,testing_labels):
    total = 0
    #total represents the total squared deviation
    count = 0
    #count represents the total number of values
    for prediction,label in zip(predictions,testing_labels):
        #loops through each prediction and each corresponding label
        diff = prediction - label
        #absolute difference
        total += diff ** 2
        #squared difference
        count += 1
    return (total/count) ** 0.5
