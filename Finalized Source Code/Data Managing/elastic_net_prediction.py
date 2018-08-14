from data_utils import extract_data_from_csv
from regression import Predictor
from datetime import datetime
from emaillib import sendEmail
import pickle

CUTOFF = 50

def get_classifier(root):
    return pickle.load(open(root+"trained_2.pkl","rb"))

def main():
    data = extract_data_from_csv("")
    predictor = Predictor(0,2,10000,distance_in_future=15)
    vectors,labels=predictor.generateVectors(data)
    predicted_temp = predictor.generatePredictions_oneValue(get_classifier(""),vectors)[-1]
    print("Predicted %d degrees Celcius (+/- 2 degrees) in 15 minutes"%predicted_temp)
    if predicted_temp > CUTOFF:
        print("Coffee should be made!")
        sendEmail("Our predictive models have determined that your machine should be turned on within the next 15 minutes.","Coffee machine should be turned on!")
    

    
main()
