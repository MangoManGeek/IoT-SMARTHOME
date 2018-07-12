import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import random

root=""
np_arr = np.genfromtxt(root+'combined.csv', delimiter=',')
data = np_arr[:,1]
data = [a if a is not None and a>0 else 0 for a in data]


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
        prediction = float(clf.predict(vector)[0])
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

def fakeify(temps,testing_portion):
    variation = 1
    for i in range(int(len(temps)*(1-testing_portion)),len(temps)):
        temps[i] *= variation
        variation += random.uniform(-0.01,0.01)
    return temps

data = fakeify(data,0.1)

clf = Ridge()
n = 25
stuff = generateVectors(n,data)
vectors = stuff[0]
labels = stuff[1]
length = len(vectors)
training_portion = 0.9
training_len = int(length * training_portion)

training_vectors = np.array(vectors[:training_len])
testing_vectors = vectors[training_len:]

testing_labels = np.array(labels[training_len:])
training_labels = np.array(labels[:training_len])

training_vectors = training_vectors.reshape(-1,n)
#testing_vectors = testing_vectors.reshape(-1,n)



plt.figure(1)
plt.subplot(211)
plt.plot(testing_labels)

print("training model")
clf.fit(training_vectors,training_labels)
print("model trained")

print("generating predictions")
predictions = generatePredictions(clf,testing_vectors,n)
print("predictions generated")

print("Cost: "+str(calculate_cost(clf,predictions,testing_labels)))

plt.subplot(212)
plt.plot(predictions)
plt.show()
    


