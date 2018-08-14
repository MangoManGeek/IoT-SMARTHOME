import smtplib
from email.mime.text import MIMEText
def sendEmail(message,subject):
    msg= MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = "winlabiot@gmail.com"
    msg['To'] = "winlabiot@gmail.com"
    s= smtplib.SMTP("smtp.gmail.com",587)
    s.ehlo()
    s.starttls()
    s.login("winlabiot","winlabiot123")
    try:
        s.sendmail("winlabiot@gmail.com",["winlabiot@gmail.com"],msg.as_string())
    except(Exception):
        print("Could not send email")
    s.quit()
