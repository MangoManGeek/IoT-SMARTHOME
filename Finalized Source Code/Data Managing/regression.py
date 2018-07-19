import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
import math
from data_utils import extract_data_from_csv
from data_utils import valid_data

# turns a list of tuples or lists into a 1D list
def flatten(arr):
    flattened = list()
    for item in arr:
        for part in item:
            flattened.append(part)
    return flattened

#generates vectors and labels based on a list of tuples and the number of preceding values that should be represented in each vector
#each vector is a list of previous values, where the corresponding label is the value that comes immediately afterwards
def generateVectors(num_previous_values,items):
    vectors = list()
    labels = list()
    for i in range(len(items)-num_previous_values-1):
        #the set of data points at index "i" are the first values of the vector
        previous_values_list = items[i:i+num_previous_values]
        vector = flatten(previous_values_list)
        #turns the list of tuples (or lists) representing the previous values into a 1D vector
        label = items[i+num_previous_values+1][0]
        #labels the vector with the next temperature value
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
        #converts the vector to a 1-sample numpy vector in order to be used for prediction
        prediction = float(clf.predict(np_vector)[0])
        predictions.append(prediction)
        if(i>0):
            if(i%num_future_values == 0):
                print("Resetting prediction frame")
                #resets the value of the vector when the given prediction length has been reached
                #happens after [num_future_values] predictions
                vector = testing_vectors[i]
            else:
                vector=vector[5:]
                #removes the first 5 values from the vector (gets rid of the first data point from use in predictions)
                vector.append(prediction)
                #adds on the predicted temperature value as the temperature value for the next vector
                for j in range(1,5):
                    vector.append(testing_vectors[i][j])
                #adds on the current OT,RH, etc to complete the next vector
                #this vector represents the state of the sensor one data point ago relative to the temperature it will be used to predict
    return predictions

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

#calculates cost values at a specific interval
def calculate_costs(bucket_size,predictions,testing_labels):
    costs = list()
    for i in range(len(predictions)):
        if(i%bucket_size == 0):
            end_index = i + bucket_size
            prediction_set = predictions[i:end_index]
            testing_label_set = testing_labels[i:end_index]
            costs.append(calculate_cost(prediction_set,testing_label_set))
    return costs

#generates fake data from a 1D list
#returns a new list with fake data
def generate_fake_list(values):
    max_val = max(values)
    min_val = min(values)
    #keeps track of the maximum and minimum values from the list to ensure that the fake data does not escape these bounds
    new_values = list()
    variation = 1
    #[variation] represents the scale factor by which values are multiplied
    #this value changes dynamically
    for i in range(len(values)):
        if(random.uniform(0,1)>0.9):
            random.seed()
            #resets the random library 10% of the time to ensure continued randomness
        new_values.append(values[i] * variation)
        if(new_values[i]>max_val):
            variation -= 0.05
        elif(new_values[i]<min_val):
            variation += 0.05
            #pushes the variation up or down depending on whether the new value has been pushed above or below the minimum
        elif(variation > 1.2):
            variation -= 0.05
        elif(variation < 0.8):
            variation += 0.05
            #pushes the variation up or down depending on whether the variation has been pushed too high or too low
        else:
            variation += random.uniform(-0.015,0.015)
            #changes the variation index by a random value
    return new_values
    
#generates fake data for a list of paired data values (e.g. a list of tuples containing temperature, light, and humidity)
#this is done by creating a separate list for each individual variable, using generate_fake_list to generate fake data for that variable, and then returning the data to its original format
def generate_fake(data_list):
    #data_list contains data as a list of x-value tuples
    sample_tuple = data_list[0]
    x = len(sample_tuple)
    array_2d = list()
    #array_2d will be a list of x lists
    for i in range(x):
        array = list()
        for item in data_list:
            array.append(item[i])
        array = generate_fake_list(array)
        #generates fake data within the array
        array_2d.append(array)
    new_array = list()
    sample_list = array_2d[0]
    list_len = len(sample_list)
    #data converted back to list of x-value pairs
    for i in range(list_len):
        paired_values = list()
        for arr in array_2d:
            paired_values.append(arr[i])
        new_array.append(paired_values)
    return new_array

#plots two 1D lists in the same figure using matplotlib
def plot(data_1, data_2):
    plt.figure(1)
    plt.plot(data_1)
    plt.plot(data_2)
    plt.show()

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

#does a basic separation of training data and testing data, and then generates vectors and labels for both the training and testing set
def generateAllData(data,trainingSize,trainingStart,testingEnd,num_previous_values,falsifyTesting):
    training_data,testing_data = separateData(data,trainingSize,trainingStart,testingEnd)
    #separates training and testing data
    if(falsifyTesting):
        testing_data = generate_fake(testing_data)
        #generates fake testing data, if falsifyTesting is set to "True"
    training_vectors, training_labels = generateVectors(num_previous_values,training_data)
    #generates training vectors and training labels from the training data
    testing_vectors,testing_labels = generateVectors(num_previous_values,testing_data)
    #generates testing vectors and testing labels from the testing data
    #for both testing and training data, vectors are generated based on the number of previous values to be used
    return training_vectors, training_labels, testing_vectors, testing_labels
    
#trains a model with vectors and labels
#the type of model that is trained depends on which "clf=" line is uncommented
def getTrainedModel(vectors, labels):
    #clf = Lasso()
    clf = ElasticNet()
    #clf = Ridge()
    clf.fit(vectors,labels)
    return clf

#returns a cost value, testing labels, and predictions given a trained model, testing data, and parameters for the vectorization and prediction
def computeCost(num_previous_values,num_future_values,falsify,trained_model,testing_data):
    if(falsify):
        testing_data = generate_fake(testing_data)
        #replaces the testing data with falsified data if falsify is set to "True"
    testing_vectors, testing_labels = generateVectors(num_previous_values,testing_data)
    #generates vectors and labels from the testing data
    print("generating predictions")
    predictions = generatePredictions(trained_model,testing_vectors,num_future_values)
    #generates predictions
    print("predictions generated")
    cost = calculate_cost(predictions,testing_labels)
    #determines the cost given the predictions and test labels
    print("Cost: "+str(cost))
    return cost,testing_labels,predictions

#generates random testing sample outside of training sample, given the proportion of the entire dataset that is to be composed of testing data
def getRandomTestingDataSample(testing_sample_proportion, training_proportion, data):
    dataset_size = len(data)
    min_testing_sample_start_index = int(dataset_size * training_proportion) + 1
    #testing sample must begin at least at the data point directly after the training data
    max_testing_sample_start_index = int(dataset_size * (1-testing_sample_proportion))
    #testing sample cannot begin any later than at the point after which it would exceed beyond the extent of the dataset
    testing_sample_start_index = random.randint(min_testing_sample_start_index, max_testing_sample_start_index)
    testing_sample_end_index = testing_sample_start_index+ int(testing_sample_proportion*dataset_size)
    return data[testing_sample_start_index:testing_sample_end_index]

#trains model on one portion of the data, tests on other part-- both falsified and unfalsified-- and prints out both costs, as well as a graph comparing actual testing labels to their predicted values
def quickComparison():
    num_previous_values = 1
    #number of previous data points represented within each vector
    num_future_values = 50000
    #number of future values to predict without looking at current temperature
    print("Gathering data")
    raw_data = extract_data_from_csv("")
    training_data, testing_data = separateData(raw_data,0.75,0,1)
    training_vectors,training_labels = generateVectors(num_previous_values,training_data)
    print("Sample vector: "+str(training_vectors[0]))
    print("Finished gathering data")    
    print("training model")
    clf = getTrainedModel(training_vectors,training_labels)
    print("model trained")
    cost_normal,testing_labels,predictions = computeCost(num_previous_values,num_future_values,False,clf,testing_data)
    cost_falsified,testing_labels_falsified,predictions_falsified = computeCost(num_previous_values,num_future_values,True,clf,testing_data)
    #incremental_costs = calculate_costs(30,predictions,testing_labels)
    #plt.figure(1)
    #plt.plot(incremental_costs)
    #plt.plot([val/30 for val in testing_labels[::30]])
    #plt.show()
    print("Cost when data is not falsified: "+str(cost_normal))
    print("Cost when data is falsified: "+str(cost_falsified))
    plot(predictions,testing_labels)

def getTemps(tuple_list):
    temps = list()
    for item in tuple_list:
        temps.append(item[0])
    return temps

def comparisonPlot():
    plt.figure(1)
    real_data= extract_data_from_csv("")
    temps = getTemps(real_data)
    print(temps[0:50])
    plt.plot(temps)
    fake_data = generate_fake(real_data)
    fake_temps = getTemps(fake_data)
    plt.plot(fake_temps)
    plt.show()

#creates a list of costs for different samples
def generateCostDistribution(num_previous,num_future,falsified,size):
    training_proportion = 0.5
    print("Gathering data")
    raw_data = extract_data_from_csv("")
    training_data = raw_data[0:int(len(raw_data)*training_proportion)]
    training_vectors,training_labels = generateVectors(num_previous,training_data)  
    print("Finished gathering data")    
    print("training model")
    clf = getTrainedModel(training_vectors,training_labels)
    print("model trained")
    costs = list()
    for i in range(size):
        testing_data = getRandomTestingDataSample(0.1,training_proportion,raw_data)        
        costs.append(computeCost(num_previous,num_future,falsified,clf,testing_data)[0])
    return costs

def main():
    costs_falsified = generateCostDistribution(1,10000,True,100)
    costs_unfalsified = generateCostDistribution(1,10000,False,100)

    total = 0
    count = 0
    for cost in costs_falsified:
        if cost>1.3:
            count +=1
        total +=1
    for cost in costs_unfalsified:
        if cost<1.3:
            count += 1
        total += 1
    print(str(count/total))
    print("Graphing distribution of cost when modified and unmodified...")
    sns.set(color_codes=True)
    sns.distplot(costs_falsified)
    sns.distplot(costs_unfalsified)
    plt.show()


main()

#comparisonPlot()

#quickComparison()

    


    


