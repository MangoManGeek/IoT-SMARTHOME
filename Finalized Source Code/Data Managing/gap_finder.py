from data_utils import datetime_object_from_text
from data_utils import datetime_object_to_seconds
from data_utils import string_from_datetime_object
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

def getDates(root):
    time_list = list()
    for line in open(root+"combined.csv"):
        date = line.split(",")[0]
        if not date == "Time":
            obj = datetime_object_from_text(date)
            seconds = datetime_object_to_seconds(obj)
            time_list.append((date,seconds))
    return time_list

def sendEmail(gap_size):
    msg= MIMEText(str(gap_size)+"-second gap detected in data")
    msg['Subject'] = "WARNING"
    msg['From'] = "orbitlab12345@gmail.com"
    msg['To'] = "dlike230@gmail.com"
    s= smtplib.SMTP("localhost")
    s.sendmail("orbitlab12345@gmail.com",["dlike230@gmail.com"],msg.as_string())
    s.quit()    

def checkForGapsAfter(lastCheckedDate,root, min_size_seconds):
    last_checked_date_as_seconds = 0 if lastCheckedDate is None else datetime_object_to_seconds(datetime_object_from_text(lastCheckedDate))
    all_dates = getDates(root)
    searchingForStartPoint = True
    log_file_output = ""
    lastDate = None
    for date_string, date_seconds in all_dates:
        if(searchingForStartPoint and date_seconds >= last_checked_date_as_seconds):
            searchingForStartPoint = False
        if(not searchingForStartPoint):
            if(lastDate is not None):
                gap_size = date_seconds - lastDate
                if(gap_size> min_size_seconds):
                    sendEmail(gap_size)
                    if(len(log_file_output) > 0):
                        log_file_output += "\n"
                    log_file_output += str(int(gap_size))+"-SECOND GAP FOUND ENDING AT " + date_string
            lastDate = date_seconds
    open("gap_log.txt","a").write(log_file_output)
    open("last_checked_gap.txt","w").write(string_from_datetime_object(datetime.now()))


checkForGapsAfter(open("last_checked_gap.txt","r").read(),"",300)
        
        
