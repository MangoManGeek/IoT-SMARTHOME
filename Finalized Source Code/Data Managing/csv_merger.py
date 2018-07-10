from os import listdir

total = ""
first = True
root = "cc2650"
for f in listdir(root):
    firstLine = True
    for line in open(root+"/"+f,"r"):
        if(firstLine):
            if(first):
                total += line
        else:
            total += line
open(root+"/combined.csv","w").write(total)
    
