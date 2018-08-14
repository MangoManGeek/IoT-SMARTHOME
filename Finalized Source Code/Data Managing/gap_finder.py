from data_utils import datetime_object_from_text
from data_utils import datetime_object_to_seconds
from data_utils import string_from_datetime_object
from datetime import datetime
from emaillib import sendEmail
import os.path
import os.remove


def getDates(root):
    time_list = list()
    for line in open(root+"combined.csv"):
        date = line.split(",")[0]
        if not date == "Time":
            obj = datetime_object_from_text(date)
            seconds = datetime_object_to_seconds(obj)
            time_list.append((date,seconds))
    return time_list

def checkForGapsAfter(lastCheckedDate,root, min_size_seconds):
    last_checked_date_as_seconds = 0 if lastCheckedDate is None else datetime_object_to_seconds(datetime_object_from_text(lastCheckedDate))
    all_dates = getDates(root)
    warning_given = os.path.isfile("important_file.txt")
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
                    sendEmail("WARNING: GAP DETECTED","A gap of size "+str(gap_size)+" was detected!")
                    if(len(log_file_output) > 0):
                        log_file_output += "\n"
                    log_file_output += str(int(gap_size))+"-SECOND GAP FOUND ENDING AT " + date_string
            lastDate = date_seconds
    if(datetime_object_to_seconds(datetime.now()) - all_dates[len(all_dates)-1][1] > min_size_seconds):
        if not warning_given:
            open("important_file.txt","w").write("DO NOT DELETE THIS FILE")
            sendEmail("WARNING: NEW DATA NOT DETECTED","New data has not been detected")
            log_file_output += "\n NO NEW DATA "+str(datetime.now())
    else:
        os.remove("important_file.txt")
    open("gap_log.txt","a").write(log_file_output)
    open("last_checked_gap.txt","w").write(string_from_datetime_object(datetime.now()))


checkForGapsAfter(open("last_checked_gap.txt","r").read(),"",120)
