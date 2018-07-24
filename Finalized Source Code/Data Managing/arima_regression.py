from statsmodels.tsa.arima_model import ARIMA
from data_utils import extract_data_from_csv
from data_utils import getDataAtIndex
from pandas import read_csv
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from data_utils import calculate_cost
from fake_data import generate_fake_list

def parser(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

def generateTimeSeriesData(root):
    data = read_csv(root+"combined.csv",header=0,parse_dates=[0],index_col=0,date_parser = parser)
    for column_name in ["OT(degC)","Humidity(RH)","Bar(millibars)","Light(lux)"]:
        del data[column_name]
    return data

series = generateTimeSeriesData("")

#takes in a list of time series data and outputs the trained model and its residuals
def train_model(series):
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit(disp=0)
    residuals = model_fit.resid
    return model_fit, residuals

#predicts future data, given trained model, and returns rmse error of prediction
def predict(trained_model,test_data):
    forecasted_data = list()
    rmse = 0
    for test_datapoint in test_data:
        forecasted_datapoint = trained_model.forecast()[0]
        forecasted_data.append(forecasted_datapoint)
        rmse += (forecasted_datapoint - test_datapoint) ** 2
    rmse /= len(test_data)
    rmse = rmse ** 0.5
    return forecasted_data, rmse

def comparison(root):
    time_series = generateTimeSeriesData(root)
    trainingEnd = int(len(time_series)/2)
    training = time_series[0:trainingEnd]
    testing = time_series["AT(degC)"].toList()[trainingEnd:]
    model_fit, residuals = train_model(training)
    forecasted_data, rmse = predict(model_fit, testing)
    rmse_fake = calculate_cost(forecasted_data,generate_fake_list(testing))
    print("Comparison between real and fake rmse:")
    print("Real: "+str(rmse))
    print("Fake: "+str(rmse_fake))
    plt.figure(1)
    plt.plot(forecasted_data)
    #plt.plot(testing)
    plt.show()
    

#plt.figure(1)
#sns.set(color_codes = True)
#sns.distplot(residuals)
#plt.show()
comparison("")
