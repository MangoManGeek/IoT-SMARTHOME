#from sklearn.linear_model import LogisticRegression
import numpy as np

root = "cc2650"
np_arr = np.genfromtxt(root+'/combined.csv', delimiter=',')
temp = np_arr[:,1]
#regression = LogisticRegression()
#regression.fit
