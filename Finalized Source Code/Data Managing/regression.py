import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
import math

#returns true if the data is a valid number
def valid_data(item):
    return item is not None and not math.isnan(item)

#takes the root directory for combined.csv as an argument, returns a tuple of corresponding values for each data entry
def extract_data_from_csv(root):
    np_arr = np.genfromtxt(root+'combined.csv', delimiter=',')
    temp_data = list()
    for time,AT,OT,RH,Bar,Light in np_arr:
        if valid_data(AT) and valid_data(OT) and valid_data(RH) and valid_data(Bar) and valid_data(Light):
            temp_data.append((AT,OT,RH,Bar,Light))
    return temp_data

# turns a list of tuples or lists into a 1D list
def flatten(arr):
    flattened = list()
    for item in arr:
        for part in item:
            flattened.append(part)
    return flattened

#generates vectors and labels based on a list of tuples and the number of preceding values that should be stored in each list
def generateVectors(num_previous_values,items):
    vectors = list()
    labels = list()
    for i in range(len(items)-num_previous_values-1):
        previous_values_list = items[i:i+num_previous_values]
        vector = flatten(previous_values_list)
        label = items[i+num_previous_values+1][0]
        vectors.append(vector)
        labels.append(label)
    return vectors,labels

#generates a given number of predictions without checking for updated AT
def generatePredictions(clf,testing_vectors,num_future_values):
    predictions = list()
    vector = testing_vectors[0]
    for i in range(len(testing_vectors)):
        np_vector = np.array(vector)
        np_vector = np_vector.reshape(1,-1)
        prediction = float(clf.predict(np_vector)[0])
        predictions.append(prediction)
        if(i>0):
            if(i%num_future_values == 0):
                #resets the value of the vector when the given prediction length has been reached
                vector = testing_vectors[i]
            else:
                vector=vector[5:]
                #removes the first 5 values from the vector (gets rid of the previous data point from use in predictions)
                vector.append(prediction)
                #adds on the predicted temperature value as the temperature value for the next vector
                for j in range(1,5):
                    vector.append(testing_vectors[i][j])
                #adds on the current OT,RH, etc to complete the vector
    return predictions

def calculate_cost(predictions,testing_labels):
    total = 0
    count = 0
    for prediction,label in zip(predictions,testing_labels):
        diff = prediction - label
        total += diff ** 2
        count += 1
    return (total/count) ** 0.5

def generate_fake_list(values):
    max_val = max(values)
    min_val = min(values)
    new_values = list()
    variation = 1
    for i in range(len(values)):
        if(random.uniform(0,1)>0.9):
            random.seed()        
        new_values.append(values[i] * variation)
        if(new_values[i]>max_val):
            variation -= 0.05
        elif(new_values[i]<min_val):
            variation += 0.05
        elif(variation > 1.2):
            variation -= 0.05
        elif(variation < 0.8):
            variation += 0.05
        else:
            variation += random.uniform(-0.015,0.015)
    return new_values
    

#note to self: fix this method
def generate_fake(data_list):
    #data_list contains data as a list of 5-value tuples
    array_2d = list()
    #array_2d will be a list of 5 lists
    for i in range(len(data_list[0])):
        array = list()
        for item in data_list:
            array.append(item[i])
        array = generate_fake_list(array)
        #generates fake data within the array
        array_2d.append(array)
    new_array = list()
    #data converted back to list of 5-value pairs
    for i in range(len(array_2d[0])):
        paired_values = list()
        for item in array_2d[i]:
            if 
    return new_array

def plot(data_1, data_2):
    plt.figure(1)
    plt.plot(data_1)
    plt.plot(data_2)
    plt.show()

def separateData(data,trainingSize,trainingStart,testingEnd):
    gap_between_training_testing = int(len(data) * 0.1)
    trainingStartIndex = int(len(data) * trainingStart)
    testingEndIndex = int(len(data) * testingEnd)
    interior_length = testingEndIndex - trainingStartIndex - gap_between_training_testing
    num_training = int(trainingSize * interior_length)
    num_testing = interior_length - trainingSize
    trainingEndIndex = trainingStartIndex + num_training
    testingStartIndex = trainingEndIndex+gap_between_training_testing
    training_data = data[trainingStartIndex:trainingEndIndex]
    testing_data = data[testingStartIndex:testingEndIndex]
    return training_data,testing_data

def generateAllData(data,trainingSize,trainingStart,testingEnd,num_previous_values,falsifyTesting):
    training_data,testing_data = separateData(data,trainingSize,trainingStart,testingEnd)
    if(falsifyTesting):
        testing_data = generate_fake(testing_data)
    training_vectors, training_labels = generateVectors(num_previous_values,training_data)
    testing_vectors,testing_labels = generateVectors(num_previous_values,testing_data)
    return training_vectors, training_labels, testing_vectors, testing_labels
    

def getTrainedModel(vectors, labels):
    #clf = Lasso()
    clf = ElasticNet()
    #clf = Ridge()
    clf.fit(vectors,labels)
    return clf

def computeCost(num_previous_values,num_future_values,falsify,trained_model,testing_data):
    if(falsify):
        testing_data = generate_fake(testing_data)
    testing_vectors, testing_labels = generateVectors(num_previous_values,testing_data)
    print("generating predictions")
    predictions = generatePredictions(trained_model,testing_vectors,num_future_values)
    print("predictions generated")
    cost = calculate_cost(predictions,testing_labels)
    print("Cost: "+str(cost))
    return cost,testing_labels,predictions

def getRandomTestingDataSample(testing_sample_proportion, training_proportion, data):
    min_testing_sample_start_index = int(len(data) * training_proportion) + 1
    max_testing_sample_start_index = int(len(data) * (1-testing_sample_proportion))
    testing_sample_start_index = random.randint(min_testing_sample_start_index, max_testing_sample_start_index)
    testing_sample_end_index = testing_sample_start_index+ int(testing_sample_proportion*len(data))
    return data[testing_sample_start_index:testing_sample_end_index]

def quickComparison():
    num_previous_values = 50
    num_future_values = 30
    print("Gathering data")
    raw_data = extract_data_from_csv("")
    training_data, testing_data = separateData(raw_data,0.75,0,1)
    training_vectors,training_labels = generateVectors(num_previous_values,training_data)
    print("Finished gathering data")    
    print("training model")
    clf = getTrainedModel(training_vectors,training_labels)
    print("model trained")
    cost_normal,testing_labels,predictions = computeCost(num_previous_values,num_future_values,False,clf,testing_data)
    cost_falsified,testing_labels,predictions_falsified = computeCost(num_previous_values,num_future_values,True,clf,testing_data)
    print("Cost when data is not falsified: "+str(cost_normal))
    print("Cost when data is falsified: "+str(cost_falsified))
    plot(predictions,testing_labels)    

def comparisonPlot():
    plt.figure(1)
    real_data= extract_data_from_csv("")
    plt.plot(getTemps(real_data))
    fake_data = generate_fake(real_data)
    plt.plot(getTemps(fake_data))
    plt.show()

def generateCostDistribution(n,falsified,size):
    training_proportion = 0.5
    print("Gathering data")
    raw_data = extract_data_from_csv("")
    training_data = raw_data[0:int(len(raw_data)*training_proportion)]
    training_vectors,training_labels = generateVectors(n,training_data)  
    print("Finished gathering data")    
    print("training model")
    clf = getTrainedModel(training_vectors,training_labels)
    print("model trained")
    costs = list()
    for i in range(size):
        testing_data = getRandomTestingDataSample(0.05,training_proportion,raw_data)        
        costs.append(computeCost(n,falsified,clf,testing_data)[0])
    return costs

def main():
    costs_falsified = generateCostDistribution(10,True,100)
    costs_unfalsified = generateCostDistribution(10,False,100)

    total = 0
    count = 0
    for cost in costs_falsified:
        if cost>5.7:
            count +=1
        total +=1
    for cost in costs_unfalsified:
        if cost<5.7:
            count += 1
        total += 1
    print(str(count/total))
    print("Graphing distribution of cost when modified and unmodified...")
    sns.set(color_codes=True)
    sns.distplot(costs_falsified)
    sns.distplot(costs_unfalsified)
    plt.show()

#loss below 5.7 = actual data

#main()

#comparisonPlot()

quickComparison()

    


    


