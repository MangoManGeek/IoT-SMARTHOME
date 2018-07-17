import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import random

def extract_data_from_csv(root):
    np_arr = np.genfromtxt(root+'combined.csv', delimiter=',')
    data = np_arr[:,1]
    data = [a if a is not None and a>0 else 0 for a in data]
    return data


def generateVectors(n,items):
    vectors = list()
    labels = list()
    for i in range(len(items)-n-1):
        vector = items[i:i+n]
        label = items[i+n+1]
        vectors.append(vector)
        labels.append(label)
    return vectors,labels

def generatePredictions(clf,testing_vectors,n):
    predictions = list()
    vector = testing_vectors[0]
    for i in range(len(testing_vectors)):
        np_vector = np.array(vector)
        np_vector = np_vector.reshape(1,-1)
        prediction = float(clf.predict(np_vector)[0])
        predictions.append(prediction)
        if(i>0):
            if(i%n == 0):
                vector = testing_vectors[i]
            else:
                vector=vector[1:]
                vector.append(prediction)
    return predictions

def calculate_cost(clf,predictions,testing_labels):
    total = 0
    count = 0
    for prediction,label in zip(predictions,testing_labels):
        diff = prediction - label
        total += diff ** 2
        count += 1
    return total/count

def generate_fake(temps,testing_portion):
    variation = 1
    for i in range(int(len(temps)*(1-testing_portion)),len(temps)):
        temps[i] *= variation
        variation += random.uniform(-0.01,0.01)
    return temps

def plot(data_1, data_2):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(data_1)
    plt.subplot(212)
    plt.plot(data_2)
    plt.show()

data = extract_data_from_csv("")

sample_size = 10000

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

def generateAllData(data,trainingSize,trainingStart,testingEnd,n,falsifyTesting):
    training_data,testing_data = separateData(data,trainingSize,trainingStart,testingEnd)
    if(falsifyTesting):
        testing_data = generate_fake(testing_data,1)
    training_vectors, training_labels = generateVectors(n,training_data)
    testing_vectors,testing_labels = generateVectors(n,testing_data)
    return training_vectors, training_labels, testing_vectors, testing_labels
    

def getTrainedModel(vectors, labels):
    clf = Ridge()
    clf.fit(vectors,labels)
    return clf

def computeCost(n,falsify):
    print("Gathering data")
    raw_data = extract_data_from_csv("")
    training_vectors,training_labels,testing_vectors,testing_labels = generateAllData(raw_data,0.8,0,1,n,falsify)
    print("Finished gathering data")
    
    print("training model")
    clf = getTrainedModel(training_vectors,training_labels)
    print("model trained")

    print("generating predictions")
    predictions = generatePredictions(clf,testing_vectors,n)
    print("predictions generated")

    cost = calculate_cost(clf,predictions,testing_labels)
    print("Cost: "+str(cost))
    return cost,testing_labels,predictions

def main():
    cost_normal,testing_labels,predictions = computeCost(10,False)
    cost_falsified,testing_labels,predictions_falsified = computeCost(10,True)
    print("Cost when data is not falsified: "+str(cost_normal))
    print("Cost when data is falsified: "+str(cost_falsified))
    plot(predictions,predictions_falsified)    

main()


    


