from datetime import datetime
import requests as req

import smtplib

import mysql.connector

#mysql constant
MYSQL_HOST='den1.mysql6.gear.host'
MYSQL_USER='winlabiot'
MYSQL_PW='winlabiot+123'
MYSQL_DB="winlabiot"
Coffee_mailing_list_table='coffee_mailing_list'


#keys in dict receive via socket
TIME='time'
AMBIENT_TEMP='ambient_temp'
OBJECT_TEMP='object_temp'

#preset values for derivative
TIME_INTERVAL=300
CURR_TIME_INTERVAL=0
MAX_DATA_POINTS=100

#openhab port and host
IP_ADDR='localhost'
PORT=8080

CURR_DERIVATIVE_URL='http://{ip}:{port}/rest/items/DataAnalyzer_CurrentDerivative'.format(port=PORT, ip=IP_ADDR)
DERIVATIVE_THRESHOLD_URL='http://{ip}:{port}/rest/items/DataAnalyzer_DerivativeThreshold'.format(port=PORT, ip=IP_ADDR)
CURR_TIME_INTERVAL_URL='http://{ip}:{port}/rest/items/DataAnalyzer_CurrentTimeInterval'.format(port=PORT, ip=IP_ADDR)

#constant to decide whether it is noise or not
#avoid keep sending email when derivative always > threshold
Making_Coffee=False
Not_Making_Coffee_Count=0

#gmail access
USER='winlabiot@gmail.com'
PASSWORD='winlabiot123'

#email info
FROM ='winlabiot@gmail.com'
TO=[]

CONTENT='Coffee will be served soon!'

def update_To_email_addr():
	#global cursor
	global TO

	#connect to GearHost mysql database
	GearHostMySQL = mysql.connector.connect(
	  host=MYSQL_HOST,
	  user=MYSQL_USER,
	  passwd=MYSQL_PW,
	  database=MYSQL_DB
	)
	cursor = GearHostMySQL.cursor()

	cursor.execute("SELECT email FROM coffee_mailing_list;")
	TO=cursor.fetchall()
	cursor.close()
	GearHostMySQL.close()





def send_email(user, password, from_addr, to_addr, content):
	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.ehlo()
	server.starttls()

	response=server.login(user,password)
	print str(datetime.now())+'			Server Response: '+str(response)
	for address in to_addr:
		server.sendmail(from_addr,address,content)
		print str(datetime.now())+'			Email Sent to '+str(address)



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

		global Making_Coffee
		global Not_Making_Coffee_Count

		self.add_data_point(newPoint)
		self.update_derivative()

		#update monitor 
		reponse=req.post(CURR_DERIVATIVE_URL, data=str(self.derivative))
		reponse=req.post(DERIVATIVE_THRESHOLD_URL, data=str(self.derivative_threshold))
		reponse=req.post(CURR_TIME_INTERVAL_URL, data=str(self.curr_time_interval))


		if(self.derivative>self.derivative_threshold):
			reponse=req.post(url, data='Making Coffee')
			if(Making_Coffee==False and Not_Making_Coffee_Count>10):
				#update target email info
				update_To_email_addr()
				send_email(USER,PASSWORD,FROM,TO,CONTENT)
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









