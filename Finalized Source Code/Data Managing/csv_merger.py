from os import listdir
from datetime import datetime
import time

def fixLine(line):
    parts = line.split(",")
    new_parts = None
    if(not parts[0]=="\n"):
        new_parts = list()
        datetime_object = datetime.strptime(parts[0], '%Y-%m-%d %H:%M:%S.%f')
        time_tuple = datetime_object.timetuple()
        absolute = time.mktime(time_tuple)
        now = time.mktime(datetime.now().timetuple())
        parts[0] = str(absolute-now)
        for i in range(len(parts)):
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
                temp = fixLine(line)
                if temp is not None:
                    if (len(temp)<num_columns):
                        temp.append(None)
                    total.append(temp)
        firstLine = False
    firstFile = False


string = ""

for n in range(len(total)):
    line = total[n]
    for i in range(len(line)):
        part = line[i]
        if part is None:
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
totalRest.sort(key = lambda x:x[0])
totalRest = totalRest[::-1]
totalRest.append(totalR1)
total = totalRest[::-1]

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
    
