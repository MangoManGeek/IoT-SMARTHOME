import socket
import thread

import data_analysis

TIME='time'
AMBIENT_TEMP='ambient_temp'
OBJECT_TEMP='object_temp'

DERIVATIVE_THRESHOLD=0.003
STATE_URL='http://{ip}:{port}/rest/items/CC2650_state'.format(port=8080, ip='localhost')

PORT=9001

def process_connnection(connection, addr):

	#create analyzer
	analyzer=data_analysis.analyzer(data_analysis.AMBIENT_TEMP,DERIVATIVE_THRESHOLD)

	while True:

		newPoint=dict()
		newPoint[TIME]=connection.recv(1024)
		newPoint[AMBIENT_TEMP]=float(connection.recv(1024))
		newPoint[OBJECT_TEMP]=float(connection.recv(1024))

		#print newPoint

		analyzer.process(newPoint, STATE_URL)

		#check if still connected
		try:
			connection.send('check')
		except:
			connection.close()
			return




def main():

	s = socket.socket()

	s.bind(('',PORT))
	print "socket binded to %s" %(PORT)

	s.listen(5) 
	print "socket is listening"

	while True:

		c, addr=s.accept()
		print ('receive connection from {conn_addr}'.format(conn_addr=str(addr)))

		try:
			thread.start_new_thread(process_connnection,(c,addr,))
		except:
			print "Error: unable to start thread"

if __name__ == '__main__':
	main()



