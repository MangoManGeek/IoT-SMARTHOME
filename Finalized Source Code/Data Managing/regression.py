import numpy as np
from sklearn.linear_model import Ridge
import random
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
import math
from data_utils import extract_data_from_csv
from data_utils import valid_data
from data_utils import getDataAtIndex
from data_utils import separateData
from fake_data import generate_fake_list
from fake_data import generate_fake
from data_utils import calculate_cost

# turns a list of tuples or lists into a 1D list
def flatten(arr):
    flattened = list()
    for item in arr:
        for part in item:
            flattened.append(part)
    return flattened


class Predictor:
    def __init__(self,index_to_predict,num_previous_values,num_future_values,distance_in_future=1):
        self.index_to_predict = index_to_predict
        self.num_previous_values = num_previous_values
        self.num_future_values = num_future_values
        self.distance_in_future = distance_in_future

    #generates vectors and labels based on a list of tuples and the number of preceding values that should be represented in each vector
    #each vector is a list of previous values, where the corresponding label is the value that comes immediately afterwards
    def generateVectors(self,items):
        vectors = list()
        labels = list()
        for i in range(len(items)-self.num_previous_values- (self.distance_in_future-1)):
            #the set of data points at index "i" are the first values of the vector
            previous_values_list = items[i:i+self.num_previous_values]
            vector = flatten(previous_values_list)
            #turns the list of tuples (or lists) representing the previous values into a 1D vector
            label = items[i+self.num_previous_values+self.distance_in_future-1][self.index_to_predict]
            #labels the vector with the next temperature value
            vectors.append(vector)
            labels.append(label)
        return vectors,labels

    def generatePredictions(self,clf,testing_vectors):
        if(self.distance_in_future == 1):
            return self.generatePredictions_multipleValues(clf,testing_vectors)
        else:
            return self.generatePredictions_oneValue(clf,testing_vectors)

    #generates a given number of predictions without checking for updated value for the predicted variable
    def generatePredictions_multipleValues(self,clf,testing_vectors):
        predictions = list()
        vector = testing_vectors[0]
        for i in range(len(testing_vectors)):
            np_vector = np.array(vector)
            np_vector = np_vector.reshape(1,-1)
            #converts the vector to a 1-sample numpy vector in order to be used for prediction
            prediction = float(clf.predict(np_vector)[0])
            predictions.append(prediction)
            if(i>0):
                if(i % self.num_future_values == 0):
                    print("Resetting prediction frame")
                    #resets the value of the vector when the given prediction length has been reached
                    #happens after [num_future_values] predictions
                    vector = testing_vectors[i]
                else:
                    vector=vector[5:]
                    lastVector = testing_vectors[i].copy()
                    lastVector[self.index_to_predict] = prediction
                    for j in range(5):
                        vector.append(lastVector[j])
                    #adds on the next vector
                    #this vector represents the state of the sensor one data point ago relative to the temperature it will be used to predict
        return predictions

    #generates single predictions
    def generatePredictions_oneValue(self,clf,testing_vectors):
        predictions = list()
        for vector in testing_vectors:
            np_vector = np.array(vector)
            np_vector = np_vector.reshape(1,-1)
            predictions.append(clf.predict(np_vector))
        return predictions
    
    #does a basic separation of training data and testing data, and then generates vectors and labels for both the training and testing set
    def generateAllData(self,data,trainingSize,trainingStart,testingEnd,falsifyTesting):
        training_data,testing_data = separateData(data,trainingSize,trainingStart,testingEnd)
        #separates training and testing data
        if(falsifyTesting):
            testing_data = generate_fake(testing_data)
            #generates fake testing data, if falsifyTesting is set to "True"
        training_vectors, training_labels = self.generateVectors(training_data)
        #generates training vectors and training labels from the training data
        testing_vectors,testing_labels = self.generateVectors(testing_data)
        #generates testing vectors and testing labels from the testing data
        #for both testing and training data, vectors are generated based on the number of previous values to be used
        return training_vectors, training_labels, testing_vectors, testing_labels

    #returns a cost value, testing labels, and predictions given a trained model, testing data, and parameters for the vectorization and prediction
    def computeCost(self,falsify,trained_model,testing_data):
        if(falsify):
            testing_data = generate_fake(testing_data)
            #replaces the testing data with falsified data if falsify is set to "True"
        testing_vectors, testing_labels = self.generateVectors(testing_data)
        #generates vectors and labels from the testing data
        print("generating predictions")
        predictions = self.generatePredictions(trained_model,testing_vectors)
        #generates predictions
        print("predictions generated")
        cost = calculate_cost(predictions,testing_labels)
        #determines the cost given the predictions and test labels
        print("Cost: "+str(cost))
        return cost,testing_labels,predictions

    #trains model on one portion of the data, tests on other part-- both falsified and unfalsified-- and prints out both costs, as well as a graph comparing actual testing labels to their predicted values
    def quickComparison(self):
        print("Gathering data")
        raw_data = extract_data_from_csv("")
        training_data, testing_data = separateData(raw_data,0.75,0,1)
        training_vectors,training_labels = self.generateVectors(training_data)
        print("Sample vector: "+str(training_vectors[0]))
        print("Finished gathering data")    
        print("training model")
        clf = getTrainedModel(training_vectors,training_labels)
        print("model trained")
        cost_normal,testing_labels,predictions = self.computeCost(False,clf,testing_data)
        cost_falsified,testing_labels_falsified,predictions_falsified = self.computeCost(True,clf,testing_data)
        #incremental_costs = calculate_costs(30,predictions,testing_labels)
        #plt.figure(1)
        #plt.plot(incremental_costs)
        #plt.plot([val/30 for val in testing_labels[::30]])
        #plt.show()
        print("Cost when data is not falsified: "+str(cost_normal))
        print("Cost when data is falsified: "+str(cost_falsified))
        return predictions,predictions_falsified,testing_labels

    #creates a list of costs for different samples
    def generateCostDistribution(self,falsified,size,num_datapoints_testing = -1):
        training_proportion = 0.5
        print("Gathering data")
        raw_data = extract_data_from_csv("")
        training_data = raw_data[0:int(len(raw_data)*training_proportion)]
        training_vectors,training_labels = self.generateVectors(training_data)  
        print("Finished gathering data")    
        print("training model")
        clf = getTrainedModel(training_vectors,training_labels)
        print("model trained")
        costs = list()
        for i in range(size):
            testing_data = getRandomTestingDataSample(0.03,training_proportion,raw_data,num_datapoints_testing)        
            costs.append(self.computeCost(falsified,clf,testing_data)[0])
        return costs

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

#trains a model with vectors and labels
#the type of model that is trained depends on which "clf=" line is uncommented
def getTrainedModel(vectors, labels):
    #clf = Lasso()
    clf = ElasticNet(alpha=0.25,l1_ratio=0.3)
    #clf = Lars()
    #clf = Ridge()
    clf.fit(vectors,labels)
    return clf

#generates random testing sample outside of training sample, given the proportion of the entire dataset that is to be composed of testing data
def getRandomTestingDataSample(testing_sample_proportion, training_proportion, data, num_datapoints_testing = -1):
    dataset_size = len(data)
    num_datapoints_testing = int(dataset_size * testing_sample_proportion) if num_datapoints_testing == -1 else num_datapoints_testing
    min_testing_sample_start_index = int(dataset_size * training_proportion) + 1
    #testing sample must begin at least at the data point directly after the training data
    max_testing_sample_start_index = dataset_size - num_datapoints_testing
    #testing sample cannot begin any later than at the point after which it would exceed beyond the extent of the dataset
    testing_sample_start_index = random.randint(min_testing_sample_start_index, max_testing_sample_start_index)
    testing_sample_end_index = testing_sample_start_index+ num_datapoints_testing
    print("Sample size: "+str(testing_sample_end_index - testing_sample_start_index))
    return data[testing_sample_start_index:testing_sample_end_index]

def accuracy_with_cutoff(real_costs,fake_costs,cutoff):
    accurate_count = 0
    total_count = 0
    for cost in real_costs:
        if cost <= cutoff:
            accurate_count += 1
        total_count += 1
    for cost in fake_costs:
        if cost > cutoff:
            accurate_count += 1
        total_count += 1
    return accurate_count/total_count

def find_threshold(real_costs,fake_costs,starting_cutoff,step):
    baseline_accuracy = accuracy_with_cutoff(real_costs,fake_costs,starting_cutoff)
    higher_cutoff = starting_cutoff + step
    lower_cutoff = starting_cutoff - step
    if(accuracy_with_cutoff(real_costs,fake_costs,lower_cutoff) > baseline_accuracy):
        return find_threshold(real_costs,fake_costs,lower_cutoff,step)
    elif(accuracy_with_cutoff(real_costs,fake_costs,higher_cutoff) > baseline_accuracy):
        return find_threshold(real_costs,fake_costs,higher_cutoff,step)
    else:
        return starting_cutoff





    


    


