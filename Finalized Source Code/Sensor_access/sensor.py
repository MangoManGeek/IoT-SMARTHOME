import math
import time
import sys
import argparse
from datetime import datetime
import os
import requests as req

from bluepy import sensortag
import bluepy
import socket


import data_analysis
import analyzer

ANALYZER_IP='localhost'

PORT=25432
IP_ADDR='10.40.4.2'

CONNECTED_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_connected'.format(port=PORT, ip=IP_ADDR)
STATE_URL='http://{ip}:{port}/rest/items/CC2650_state'.format(port=PORT, ip=IP_ADDR)
AMBIENT_TEMP_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_ambient_temp'.format(port=PORT, ip=IP_ADDR)
OBJECT_TEMP_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_object_temp'.format(port=PORT, ip=IP_ADDR)
HUMIDITY_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_humidity'.format(port=PORT, ip=IP_ADDR)
LIGHT_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_light'.format(port=PORT, ip=IP_ADDR)
BAROMETER_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_barometer'.format(port=PORT, ip=IP_ADDR)
ACCELEROMETERX_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_accelerometerx'.format(port=PORT, ip=IP_ADDR)
ACCELEROMETERY_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_accelerometery'.format(port=PORT, ip=IP_ADDR)
ACCELEROMETERZ_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_accelerometerz'.format(port=PORT, ip=IP_ADDR)
MAGNETOMETERX_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_magnetometerx'.format(port=PORT, ip=IP_ADDR)
MAGNETOMETERY_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_magnetometery'.format(port=PORT, ip=IP_ADDR)
MAGNETOMETERZ_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_magnetometerz'.format(port=PORT, ip=IP_ADDR)
GYROSCOPEX_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_gyroscopex'.format(port=PORT, ip=IP_ADDR)
GYROSCOPEY_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_gyroscopey'.format(port=PORT, ip=IP_ADDR)
GYROSCOPEZ_URL='http://{ip}:{port}/rest/items/bluetooth_CC2650_01_gyroscopez'.format(port=PORT, ip=IP_ADDR)

'''
CONNECTED_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_connected'
AMBIENT_TEMP_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_ambient_temp'
OBJECT_TEMP_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_object_temp'
HUMIDITY_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_humidity'
LIGHT_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_light'
BAROMETER_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_barometer'
ACCELEROMETERX_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_accelerometerx'
ACCELEROMETERY_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_accelerometery'
ACCELEROMETERZ_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_accelerometerz'
MAGNETOMETERX_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_magnetometerx'
MAGNETOMETERY_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_magnetometery'
MAGNETOMETERZ_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_magnetometerz'
GYROSCOPEX_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_gyroscopex'
GYROSCOPEY_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_gyroscopey'
GYROSCOPEZ_URL='http://localhost:25432/rest/items/bluetooth_CC2650_01_gyroscopez'
'''

DERIVATIVE_THRESHOLD=0.003

def enableSensors(arg, tag):

	# Enabling selected sensors

	if arg.temperature or arg.all:
		tag.IRtemperature.enable()
		print (str(datetime.now())+'			'+'IRtemperature Sensor Enabled')
	if arg.humidity or arg.all:
		tag.humidity.enable()
		print (str(datetime.now())+'			'+'Humidity Sensor Enabled')
	if arg.barometer or arg.all:
		tag.barometer.enable()
		print (str(datetime.now())+'			'+'Barometer Sensor Enabled')
	if arg.accelerometer or arg.all:
		tag.accelerometer.enable()
		print (str(datetime.now())+'			'+'Accelerometer Sensor Enabled')
	if arg.magnetometer or arg.all:
		tag.magnetometer.enable()
		print (str(datetime.now())+'			'+'Magnetometer Sensor Enabled')
	if arg.gyroscope or arg.all:
		tag.gyroscope.enable()
		print (str(datetime.now())+'			'+'Gyroscope Sensor Enabled')
	if arg.keypress or arg.all:
		tag.keypress.enable()
		tag.setDelegate(sensortag.KeypressDelegate())
		print (str(datetime.now())+'			'+'Keypress Sensor Enabled')
	if arg.light and tag.lightmeter is None:
		print (str(datetime.now())+'			'+'Warning: no light sensor on this device')
	if (arg.light or arg.all) and tag.lightmeter is not None:
		tag.lightmeter.enable()
		print (str(datetime.now())+'			'+'Light Sensor Enabled')


def write_csv_header(file, arg, tag):
	#write csv headers
	header_buffer=''
	header_buffer+=('Time,')
	if arg.temperature or arg.all:
		header_buffer+=('Ambient Temperature (degC),')
		header_buffer+=('Object Temperature (degC),')
	if arg.humidity or arg.all:
		header_buffer+=('Humidity (RH),')
	if arg.barometer or arg.all:
		header_buffer+=('Barometer (millibars),')
	if arg.accelerometer or arg.all:
		header_buffer+=('Accelerometer-x (g),')
		header_buffer+=('Accelerometer-y (g),')
		header_buffer+=('Accelerometer-z (g),')
	if arg.magnetometer or arg.all:
		header_buffer+=('Magnetometer-x (uT),')
		header_buffer+=('Magnetometer-y (uT),')
		header_buffer+=('Magnetometer-z (uT),')
	if arg.gyroscope or arg.all:
		header_buffer+=('Gyroscope-x (deg/sec),')
		header_buffer+=('Gyroscope-y (deg/sec),')
		header_buffer+=('Gyroscope-z (deg/sec),')
	if (arg.light or arg.all) and tag.lightmeter is not None:
		header_buffer+=('Light (lux),')

	file.write(header_buffer[:-1])
	file.write('\n')


#use to update file descriptor 
#make a csv file everyday
def update_file_des(file, arg, tag):
	
	n=datetime.now()

	filename=arg.filepath+str(n.year)+'-'+str(n.month)+'-'+str(n.day)+'_raw_data.csv'
	
	if(filename==file.name):
		#date has not been changed
		#return original file descriptor
		return file
	else:
		#close previous file descriptor 
		file.close()
		new_file=open(filename, 'a')
		print (str(datetime.now())+'			'+'File '+filename+' Created')
		write_csv_header(new_file, arg, tag)

		return new_file




def main():


	#add parse arguement
	parser = argparse.ArgumentParser()
	parser.add_argument('host', action='store',help='MAC of BT device')
	path_to_curr=os.path.dirname(os.path.abspath(__file__))+'/'
	parser.add_argument('-f', '--filepath', action='store',help='path/to/output_csv_file', default=path_to_curr)
	#parser.add_argument('-n', action='store', dest='count', default=0, type=int, help="Number of times to loop data")
	parser.add_argument('-t',action='store',type=float, default=1.0, help='time between polling')
	parser.add_argument('-T','--temperature', action="store_true",default=False)
	parser.add_argument('-A','--accelerometer', action='store_true', default=False)
	parser.add_argument('-H','--humidity', action='store_true', default=False)
	parser.add_argument('-M','--magnetometer', action='store_true', default=False)
	parser.add_argument('-B','--barometer', action='store_true', default=False)
	parser.add_argument('-G','--gyroscope', action='store_true', default=False)
	parser.add_argument('-K','--keypress', action='store_true', default=False)
	parser.add_argument('-L','--light', action='store_true', default=False)
	parser.add_argument('--all', action='store_true', default=False)

	#parse arguments
	arg = parser.parse_args(sys.argv[1:])

	print (str(datetime.now())+'			'+'Starting...')
	time.sleep(3.0)	

	#connect to sensor tag
	print(str(datetime.now())+'			'+'Connecting to ' + arg.host)
	tag = sensortag.SensorTag(arg.host)

	print (str(datetime.now())+'			'+'Connection Successful')


	enableSensors(arg, tag);

	sys.stdout.flush()




	#wait for sensor initialization
	time.sleep(1.0)
	n=datetime.now()

	print(str(datetime.now())+'			'+'Start Measuring...')

	#store pid
	#pid_filename=arg.filepath+str(n.year)+'-'+str(n.month)+'-'+str(n.day)+'_raw_data.pid'
	pid_filename=arg.filepath+'sensor.pid'
	pid_file=open(pid_filename, 'w')
	pid_file.write(str(os.getpid()))
	pid_file.close()



	#create new csv file
	filename=arg.filepath+str(n.year)+'-'+str(n.month)+'-'+str(n.day)+'_raw_data.csv'
	file=open(filename, 'a')

	'''
	#write csv headers
	header_buffer=''
	header_buffer+=('Time,')
	if arg.temperature or arg.all:
		header_buffer+=('Ambient Temperature (degC),')
		header_buffer+=('Object Temperature (degC),')
	if arg.humidity or arg.all:
		header_buffer+=('Humidity (RH),')
	if arg.barometer or arg.all:
		header_buffer+=('Barometer (millibars),')
	if arg.accelerometer or arg.all:
		header_buffer+=('Accelerometer-x (g),')
		header_buffer+=('Accelerometer-y (g),')
		header_buffer+=('Accelerometer-z (g),')
	if arg.magnetometer or arg.all:
		header_buffer+=('Magnetometer-x (uT),')
		header_buffer+=('Magnetometer-y (uT),')
		header_buffer+=('Magnetometer-z (uT),')
	if arg.gyroscope or arg.all:
		header_buffer+=('Gyroscope-x (deg/sec),')
		header_buffer+=('Gyroscope-y (deg/sec),')
		header_buffer+=('Gyroscope-z (deg/sec),')
	if (arg.light or arg.all) and tag.lightmeter is not None:
		header_buffer+=('Light (lux),')

	file.write(header_buffer[:-1])
	file.write('\n')
	'''
	if os.stat(filename).st_size ==0:
		#first time create file
		print (str(datetime.now())+'			'+'Writing CSV Header...')
		write_csv_header(file, arg, tag)

	#create connection to analyzer server
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.connect((ANALYZER_IP, analyzer.PORT))


	while True:

		file=update_file_des(file, arg, tag)

		line_buffer=''

		try:
			#file.write(str(datetime.now())+',')

			#Update connected
			reponse=req.post(CONNECTED_URL,data='ON')

			line_buffer+=(str(datetime.now())+',')

			#create new dataPoint for analysis
			newPoint=dict()


			if arg.temperature or arg.all:
				#file.write(str(tag.IRtemperature.read()[0])+',')
				#file.write(str(tag.IRtemperature.read()[1])+',')
				line_buffer+=(str(tag.IRtemperature.read()[0])+',')
				line_buffer+=(str(tag.IRtemperature.read()[1])+',')

				try:
					s.sendall(str(datetime.now()))
					s.sendall(str(tag.IRtemperature.read()[0]))
					s.sendall(str(tag.IRtemperature.read()[1]))
				except:
					#reconnect
					s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
					s.connect((ANALYZER_IP, analyzer.PORT))


				reponse=req.post(AMBIENT_TEMP_URL, data=str(round(tag.IRtemperature.read()[0],1)))
				reponse=req.post(OBJECT_TEMP_URL, data=str(round(tag.IRtemperature.read()[1],1)))

			if arg.humidity or arg.all:
				#file.write(str(tag.humidity.read()[1])+',')
				line_buffer+=(str(tag.humidity.read()[1])+',')

				reponse=req.post(HUMIDITY_URL, data=str(round(tag.humidity.read()[1],1)))

			if arg.barometer or arg.all:
				#file.write(str(tag.barometer.read()[1])+',')
				line_buffer+=(str(tag.barometer.read()[1])+',')

				reponse=req.post(BAROMETER_URL, data=str(round(tag.barometer.read()[1],1)))

			if arg.accelerometer or arg.all:
				#file.write(str(tag.accelerometer.read()[0])+',')
				#file.write(str(tag.accelerometer.read()[1])+',')
				#file.write(str(tag.accelerometer.read()[2])+',')
				line_buffer+=(str(tag.accelerometer.read()[0])+',')
				line_buffer+=(str(tag.accelerometer.read()[1])+',')
				line_buffer+=(str(tag.accelerometer.read()[2])+',')

				reponse=req.post(ACCELEROMETERX_URL, data=str(round(tag.accelerometer.read()[0],1)))
				reponse=req.post(ACCELEROMETERY_URL, data=str(round(tag.accelerometer.read()[1],1)))
				reponse=req.post(ACCELEROMETERZ_URL, data=str(round(tag.accelerometer.read()[2],1)))

			if arg.magnetometer or arg.all:
				line_buffer+=(str(tag.magnetometer.read()[0])+',')
				line_buffer+=(str(tag.magnetometer.read()[1])+',')
				line_buffer+=(str(tag.magnetometer.read()[2])+',')

				reponse=req.post(MAGNETOMETERX_URL, data=str(round(tag.magnetometer.read()[0],1)))
				reponse=req.post(MAGNETOMETERY_URL, data=str(round(tag.magnetometer.read()[1],1)))
				reponse=req.post(MAGNETOMETERZ_URL, data=str(round(tag.magnetometer.read()[2],1)))

			if arg.gyroscope or arg.all:
				line_buffer+=(str(tag.gyroscope.read()[0])+',')
				line_buffer+=(str(tag.gyroscope.read()[1])+',')
				line_buffer+=(str(tag.gyroscope.read()[2])+',')

				reponse=req.post(GYROSCOPEX_URL, data=str(round(tag.gyroscope.read()[0],1)))
				reponse=req.post(GYROSCOPEY_URL, data=str(round(tag.gyroscope.read()[1],1)))
				reponse=req.post(GYROSCOPEZ_URL, data=str(round(tag.gyroscope.read()[2],1)))

			if (arg.light or arg.all) and tag.lightmeter is not None:
				line_buffer+=(str(tag.lightmeter.read())+',')

				reponse=req.post(LIGHT_URL, data=str(round(tag.lightmeter.read(),1)))

			file.write(line_buffer[:-1])
			file.write('\n')

			file.flush()


			tag.waitForNotifications(arg.t)
		except bluepy.btle.BTLEException as exception:

			#Update connected
			reponse=req.post(CONNECTED_URL,data='OFF')


			print (str(datetime.now())+'			'+str(type(exception).__name__) + 'caught')
			print (str(datetime.now())+'			'+'RECONNECTING...')
			tag = sensortag.SensorTag(arg.host)
			enableSensors(arg, tag)

		#file.flush()
		#tag.waitForNotifications(arg.t)
	file.close()
	tag.disconnect()
	




if __name__ == '__main__':
	main()


    













