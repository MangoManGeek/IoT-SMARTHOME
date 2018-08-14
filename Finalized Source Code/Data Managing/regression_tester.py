from regression import Predictor
from graphlib import distPlot
from graphlib import plot
from graphlib import comparisonPlot
import pickle
from regression import find_threshold
from regression import accuracy_with_cutoff
from data_utils import extract_data_from_csv
from regression import getTrainedModel



def main_1():
    predictor = Predictor(0,2,10000,distance_in_future=15)
    costs_falsified = predictor.generateCostDistribution(True,50,num_datapoints_testing = 30)
    costs_unfalsified = predictor.generateCostDistribution(False,50,num_datapoints_testing = 30)

    cutoff = find_threshold(costs_unfalsified,costs_falsified,18,0.025)
    print("Ideal loss cutoff: "+str(cutoff))
    print("Accuracy of prediction: "+str(accuracy_with_cutoff(costs_unfalsified,costs_falsified,cutoff)))
    
    print("Graphing distribution of cost when modified and unmodified...")
    distPlot(costs_falsified,costs_unfalsified)

def main_2():
    predictor = Predictor(0,2,10000,distance_in_future=15)
    predictor.quickComparison()

def store_training():
    predictor = Predictor(0,2,100000,distance_in_future=15)
    vectors, labels = predictor.generateVectors(extract_data_from_csv(""))
    trained_model = getTrainedModel(vectors,labels)
    file = open("trained.pkl","wb")
    pickle.dump(trained_model,file,protocol = 0)

def store_training_2():
    predictor = Predictor(0,2,10000,distance_in_future=15)
    vectors, labels = predictor.generateVectors(extract_data_from_csv(""))
    trained_model = getTrainedModel(vectors,labels)
    file = open("trained_2.pkl","wb")
    pickle.dump(trained_model,file,protocol = 0)

#main_1()

#comparisonPlot()

#store_training()

store_training_2()
