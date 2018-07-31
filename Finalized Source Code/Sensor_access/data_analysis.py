from datetime import datetime
import requests as req

import smtplib

TIME='time'
AMBIENT_TEMP='ambient_temp'
OBJECT_TEMP='object_temp'

TIME_INTERVAL=300
CURR_TIME_INTERVAL=0
MAX_DATA_POINTS=100

IP_ADDR='localhost'
PORT=8080

CURR_DERIVATIVE_URL='http://{ip}:{port}/rest/items/DataAnalyzer_CurrentDerivative'.format(port=PORT, ip=IP_ADDR)
DERIVATIVE_THRESHOLD_URL='http://{ip}:{port}/rest/items/DataAnalyzer_DerivativeThreshold'.format(port=PORT, ip=IP_ADDR)
CURR_TIME_INTERVAL_URL='http://{ip}:{port}/rest/items/DataAnalyzer_CurrentTimeInterval'.format(port=PORT, ip=IP_ADDR)

#constant to decide whether it is noise or not
#avoid keep sending email when derivative always > threshold
Making_Coffee=False
Not_Making_Coffee_Count=0


USER='winlabiot@gmail.com'
PASSWORD='winlabiot123'

FROM ='winlabiot@gmail.com'
TO='1165268303@qq.com'

CONTENT='Coffee will be served soon!'


def send_email(user, password, from_addr, to_addr, content):
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.ehlo()
	server.starttls()

	response=server.login(user,password)
	print 'Server Response: '+str(response)
	server.sendmail(from_addr,to_addr,content)
	print 'email sent'



class analyzer:

	#data type is AMBIENT_TEMP or OBJECT_TEMP
	#derivative_threshold is degree/sec
	def __init__(self,data_type,derivative_threshold, max_data_points=100,time_interval=300):
		#data is array of dict data points
		self.data=[]
		#start index is the earliest data point
		self.start_index=0
		self.derivative=0
		self.time_interval=time_interval
		self.curr_time_interval=0
		self.key=data_type
		self.max_data_points=max_data_points
		self.derivative_threshold=derivative_threshold

	def process(self,newPoint, url):
		self.add_data_point(newPoint)
		self.update_derivative()

		#update monitor 
		reponse=req.post(CURR_DERIVATIVE_URL, data=str(self.derivative))
		reponse=req.post(DERIVATIVE_THRESHOLD_URL, data=str(self.derivative_threshold))
		reponse=req.post(CURR_TIME_INTERVAL_URL, data=str(self.curr_time_interval))


		if(self.derivative>self.derivative_threshold):
			reponse=req.post(url, data='Making Coffee')
			if(Making_Coffee==False and Not_Making_Coffee_Count>10):
				send_email(USER,PASSWORD,TO,FROM,CONTENT)
			
			#update constant
			Making_Coffee=True 
			Not_Making_Coffee_Count=0
		else:
			reponse=req.post(url, data='Not Ready')	
			
			#update constant
			Making_Coffee=False
			Not_Making_Coffee_Count+=1



	#data --> dict
	def add_data_point(self,newPoint):

		newPoint[TIME]=self.str2datetime(newPoint[TIME])
		self.data.append(newPoint)
		self.curr_time_interval=(self.data[len(self.data)-1][TIME]-self.data[self.start_index][TIME]).total_seconds()

		#clear expired date if max data points is reached
		if(len(self.data)>self.max_data_points):
			del self.data[0:start_index]
			start_index=0



		'''		
		if (len(self.data)==5):
			#replace expired data point
			self.data[self.start_index]=newPoint
			#update start index
			if self.start_index==4:
				self.start_index=0
			else:
				self.start_index+=1
		else:
			self.data.append(newPoint)
		'''

	def str2datetime(self, datetime_string):
		return datetime.strptime(datetime_string, '%Y-%m-%d %H:%M:%S.%f')


	def update_derivative(self):
		if self.curr_time_interval<self.time_interval:
			return
		else:
			self.derivative=1.0*(self.data[len(self.data)-1][self.key]-self.data[self.start_index][self.key])/self.curr_time_interval
			#update start_index
			self.start_index+=1

			#update curr_time_interval
			self.curr_time_interval=(self.data[len(self.data)-1][TIME]-self.data[self.start_index][TIME]).total_seconds()









