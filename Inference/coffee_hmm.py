import numpy as np 
import csv

import data_analysis

DERIVATIVE_THRESHOLD=0.03

EMPTY=1
NOT_EMPTY=0

CSV_TIME='Time'
CSV_LIGHT='Light (lux)'
#Object_t='Object Temperature (degC)'
#HUMIDITY='Humidity (RH)'
CSV_AMBIENT_T='Ambient Temperature (degC)'

def isPotRemoved(last, curr):
	if last<=10 and curr >= 20:
		return True
	else:
		return False

class Coffee_HMM:

	def __init__(self):
		self.transition_matrix=np.matrix([[0,0],[0,0]])
		self.observation_matrices=dict()
		self.max_pot_removal_number=0

		#[p(Empty),p(not Empty)] --> pre-set priors 
		self.p=[0.1,0.9]

		#training parameters

		#used as denominator when calculate P(p=E|p=not E)
		self.num_of_coffee_making_cycles=0

		#average number of pot removal within one cycle
		self.avg_num_of_pot_removal=0
		
		self.num_of_pot_empty=0
		self.num_of_pot_not_empty=0

		#number of r>=n|pot Empty  --> use n as key in dict
		self.x=dict()
		#number of r>=n|pot not Empty   --> use n as key in dict
		self.y=dict()

	
	#pot states:array of 0s and 1s where 0 is not empty, 1 is empty
	#assume length of both lists are same
	#assume number_of_pot_removals starts from 1
	def trainCSV(self,csvName):
		file=open(csvName,'r')
		reader=csv.DictReader(file)
		light=[]
		temp=[]
		time=[]
		for row in reader:
			light.append(float(row[CSV_LIGHT]))
			temp.append(float(row[CSV_AMBIENT_T]))
			time.append(row[CSV_TIME])

		analyzer=data_analysis.analyzer(data_analysis.AMBIENT_TEMP,DERIVATIVE_THRESHOLD)

		pot_remove_count=1
		pot_remove_arr=[]
		state_arr=[]
		#state=NOT_EMPTY

		for i in range(1,len(time)):
			newPoint=dict()
			newPoint[data_analysis.TIME]=time[i]
			newPoint[data_analysis.AMBIENT_TEMP]=temp[i]
			if analyzer.process(newPoint,url=None,update_monitor=False,email=False):
				#if making coffee
				#reset pot remove
				pot_remove=1
				#change state of last data point to empty
				state_arr[len(state_arr)-1]=EMPTY

			if isPotRemoved(light[i-1], light[i]):
				pot_remove_arr.append(pot_remove_count)
				state_arr.append(NOT_EMPTY)

				pot_remove_count+=1

		#return pot_remove_arr,state_arr

		self.train(state_arr, pot_remove_arr)

			



	def train(self,pot_states, number_of_pot_removals):

		if len(pot_states)!=len(number_of_pot_removals) :
			raise Exception('expect length of lists are the same')

		self.max_pot_removal_number=max(number_of_pot_removals)	
		#initialize x and y
		for i in range(1,self.max_pot_removal_number+1):
			self.x[i]=0
			self.y[i]=0

		

		removals=[]
		for i in range(len(pot_states)):
			
			if pot_states[i]==EMPTY:
				removals.append(number_of_pot_removals[i])

		if len(removals)==0:
			self.avg_num_of_pot_removal=1.0*number_of_pot_removals[len(number_of_pot_removals)-1]/1
		else:
			self.avg_num_of_pot_removal=1.0*sum(removals)/len(removals)





		#number of pot emptys is number of cycles since we record states when pot is been removed.
		self.num_of_coffee_making_cycles=pot_states.count(EMPTY)
		self.num_of_pot_empty=pot_states.count(EMPTY)

		self.num_of_pot_not_empty=pot_states.count(NOT_EMPTY)

		for i in range(len(pot_states)):
			#for each removal number
			for j in range(1,self.max_pot_removal_number+1):

				if number_of_pot_removals[i]>=j and pot_states[i]==EMPTY:
					self.x[j]+=1
				if number_of_pot_removals[i]>=j and pot_states[i]==NOT_EMPTY:
					self.y[j]+=1

		#transition matrix
		#p(Empty|Empty)
		self.transition_matrix[0,0]=0
		#p(not Empty|Empty)
		self.transition_matrix[0,1]=1-self.transition_matrix[0,0]
		#p(Empty|not Empty)
		self.transition_matrix[1,0]=1.0/self.avg_num_of_pot_removal
		#p(not Empty|not Empty)
		self.transition_matrix[1,1]=1-self.transition_matrix[1,0]



		#observation matrices
		for i in range(1,self.max_pot_removal_number+1):
			#here change to avoid both self.num_of_pot_empty and self.num_of_pot_not_empty =0
			#right now is just for test purposes
			if self.num_of_pot_empty==0:
				self.observation_matrices[i]=np.matrix([[1.0*self.x[i]/1,0],[0,1.0*self.y[i]/self.num_of_pot_not_empty]])
			else:
				self.observation_matrices[i]=np.matrix([[1.0*self.x[i]/self.num_of_pot_empty,0],[0,1.0*self.y[i]/self.num_of_pot_not_empty]])


	#num_of_pot_remove is the new observation
	def predict(self,num_of_pot_remove):
		alpha=pow(self.p*self.transition_matrix*self.observation_matrices[num_of_pot_remove]*np.matrix([[1],[1]]),-1)
		self.p=alpha[0][0]*self.p*self.transition_matrix*self.observation_matrices[num_of_pot_remove]
		'''
		#update training data as predict
		if num_of_pot_remove > self.max_pot_removal_number:
			self.max_pot_removal_number=num_of_pot_remove
		'''

		return self.p


if __name__ == '__main__':
		
	h=Coffee_HMM()
	h.trainCSV('try.csv')
	print h.predict(2)











