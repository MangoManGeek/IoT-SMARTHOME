from data_utils import extract_data_from_csv
from regression import Predictor
from datetime import datetime
from emaillib import sendEmail
import pickle

def get_classifier(root):
    return pickle.load(open(root+"trained.pkl","rb"))

def get_last_window(num_minutes,root):
    data = extract_data_from_csv(root)
    data = data[(len(data)-num_minutes):(len(data))]
    return data

def check_window(num_minutes,root):
    classifier = get_classifier(root)
    window = get_last_window(num_minutes,root)
    predictor = Predictor(0,2,10000,distance_in_future=15)
    cost = predictor.computeCost(False,classifier,window)[0]
    if(cost > 18):
        open("falsified_log.txt","a").write("\n"+str(datetime.now())+" SUSPICIOUS DATA")
        sendEmail("Suspicious data detected!","Please check your device for malfunction or report an attack.")        
    else:
        open("falsified_log.txt","a").write("\n"+str(datetime.now())+" Data looks good")



check_window(30,"")
