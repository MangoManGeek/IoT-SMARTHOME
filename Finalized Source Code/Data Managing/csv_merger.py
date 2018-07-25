from os import listdir
from datetime import datetime
import time

def fixLine(line):
    parts = line.strip("\n").split(",")
    new_parts = None
    if(not parts[0]=="\n"):
        new_parts = list()
        datetime_object = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S.%f')
        time_tuple = datetime_object.timetuple()
        parts[0] = str(time_tuple[0])+"-"+str(time_tuple[1])+"-"+str(time_tuple[2])+" "+str(time_tuple[3])+":"+str(time_tuple[4])+":"+str(time_tuple[5])
        new_parts.append(parts[0])
        for i in range(1,len(parts)):
            part = parts[i]
            if len(part)<1:
                new_parts.append(None)
            else:
                new_parts.append(float(part))
    return new_parts

total = list()
firstFile = True
root = "cc2650/"
num_columns = 0
for f in listdir(root):
    firstLine = True
    for line in open(root+f,"r"):
        line = line.strip("\n")
        if "Time" in line:
            firstLine = True
        if(len(line)>0):
            if(firstLine):
                if(firstFile):
                    flp = line.split(",")
                    total.append(flp)
                    num_columns = len(flp)
                else:
                    print("Removed: "+line)
            else:
                temp = fixLine(line)
                if temp is not None:
                    if (len(temp)<num_columns):
                        temp.append(None)
                    total.append(temp)
                else:
                    print("Removed: "+line)
        firstLine = False
    firstFile = False

for n in range(len(total)):
    light_backup = total[n][14]
    total[n] = total[n][:5]
    total[n].append(light_backup)

string = ""

for n in range(len(total)):
    line = total[n]
    for i in range(len(line)):
        part = line[i]
        if part is None and i>0:
            startIndex = n-1
            stopIndex = n+1
            while total[stopIndex][i] is None:
                stopIndex += 1
            while total[startIndex][i] is None:
                startIndex -= 1
            print((startIndex,stopIndex))
            line[i] = (total[stopIndex][i]+total[startIndex][i])/2

totalR1 = total[0]
totalRest = total[1:]
totalRest.sort(key = lambda x:time.mktime(datetime.strptime(x[0], '%Y-%m-%d %H:%M:%S').timetuple()))
totalRest = totalRest[::-1]
totalRest.append(totalR1)
total = totalRest[::-1]
total[0][0]= "Time"
total[0][1]= "AT(degC)"
total[0][2]= "OT(degC)"
total[0][3]= "Humidity(RH)"
total[0][4]= "Bar(millibars)"
total[0][5]= "Light(lux)"

for line in total:
    lineText = ""
    for part in line:
        if(len(lineText)>0):
            lineText += ","
        lineText += str(part)
    if(len(string)>0):
        string +="\n"
    string += lineText    
        
open(root.strip("cc2650/")+"combined.csv","w").write(string)
    
