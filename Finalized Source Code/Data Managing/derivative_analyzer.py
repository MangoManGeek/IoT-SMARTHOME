from data_utils import extract_data_from_csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def computeDerivatives(data):
    result = list()
    num_dimensions = len(data[0])
    for dimension in range(num_dimensions):
        result.append(list())
    last = None
    for item in data:
        if(last is not None):
            for i in range(len(item)):
                result[i].append(item[i]-last[i])
        last = item
    return result

def computeConfidenceIntervals(lists):
    intervals = list()
    for arr in lists:
        mean = sum(arr)/len(arr)
        stdev = np.std(arr)
        margin = stdev*6
        intervals.append((mean-margin,mean+margin))
    return intervals

def list_ranges(derivatives):
    ranges = list()
    for derivative_list in derivatives:
        ranges.append(max(derivative_list) - min(derivative_list))
    return ranges

def findSuspicious(derivatives,confidenceIntervals):
    suspicious = dict()
    for i in range(len(derivatives)):
        interval = confidenceIntervals[i]
        for n in range(len(derivatives[i])):
            data = derivatives[i][n]
            if(data < interval[0] or data > interval[1]):
                suspicious[n]=True
    return len(suspicious.keys())

def main():
    data = extract_data_from_csv("")    
    derivatives = computeDerivatives(data)
    """sns.set(color_codes=True)
    sns.distplot(derivatives[4])
    plt.show()"""
    intervals = computeConfidenceIntervals(derivatives)
    print(intervals)
    print(findSuspicious(derivatives,intervals))
    

main()
    

