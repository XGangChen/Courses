from gpiozero import Button
from signal import pause
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# Email credentials
EMAIL_ADDRESS = "vincent13887@gmail.com"
EMAIL_PASSWORD = "mdmlzyfsgaijixxr"  # Use the Gmail App Password
EMAIL_TO = "xiaogang9432@gmail.com"

def send_email():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = MIMEText(f"? The button was pressed on Raspberry Pi at {current_time}")
    msg["Subject"] = f"Button Press Alert ? {current_time}"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_TO

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"? Email sent at {current_time}")
    except Exception as e:
        print(f"? Failed to send email at {current_time}: {e}")

# Setup GPIO button (on GPIO 17)
button = Button(17)

# Trigger the email every time the button is pressed
button.when_pressed = send_email

print("Waiting for button presses... (Press Ctrl+C to exit)")
pause()
