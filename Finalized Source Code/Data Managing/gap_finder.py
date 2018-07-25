from data_utils import datetime_object_from_text
from data_utils import datetime_object_to_seconds
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
    searchingForStartPoint = True
    lastDate = None
    for date_string, date_seconds in all_dates:
        if(searchingForStartPoint and date_seconds >= last_checked_date_as_seconds):
            searchingForStartPoint = False
        if(lastDate is not None and date_seconds - lastDate > min_size_seconds):
            print("GAP FOUND AT " + date_string)
        lastDate = date_seconds


checkForGapsAfter(None,"",120)
        
        
