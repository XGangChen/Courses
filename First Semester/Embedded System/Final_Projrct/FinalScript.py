import cv2
import time
import os
from gpiozero import MotionSensor
from ultralytics import YOLO
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.legacy import show_message
from luma.core.legacy.font import proportional, LCD_FONT
from threading import Thread
import smtplib
from email.message import EmailMessage

# --- Email alert settings ---
EMAIL_FROM = 'vincent13887@gmail.com'
EMAIL_TO = 'xiaogang9432@gmail.com'
EMAIL_PASS = 'mdmlzyfsgaijixxr'  # Use app password for Gmail

def send_email_alert(image_path):
    msg = EmailMessage()
    msg['Subject'] = 'Door Alert: Person Detected'
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    msg.set_content('A person was detected at the door. See attached snapshot.')

    with open(image_path, 'rb') as f:
        img_data = f.read()
        msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_FROM, EMAIL_PASS)
        smtp.send_message(msg)
        print(f"WARNING email sent to {EMAIL_TO}")

def send_clear_email(image_path):
    msg = EmailMessage()
    msg['Subject'] = 'Door Clear: Person Left'
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    msg.set_content('The area is now clear. Snapshot before clearance is attached.')

    if image_path and os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpeg', filename=os.path.basename(image_path))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_FROM, EMAIL_PASS)
        smtp.send_message(msg)
        print(f"CLEAR email sent to {EMAIL_TO}")

# --- Prepare folder for snapshots ---
os.makedirs("snapshots", exist_ok=True)

# --- Initialize YOLOv11 model ---
model = YOLO('yolo11n.pt')

# --- Initialize Camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# --- Initialize MAX7219 LED matrix ---
serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=1, block_orientation=0)

# --- Initialize PIR sensor ---
pir = MotionSensor(17)

# --- Detection Tracking ---
DETECTION_DURATION = 10  # seconds of continuous person detection
WARNING_CLEAR_DELAY = 3  # seconds of no person to show "CLEAR"

detection_start_time = None
no_person_start_time = None
warning_active = False
stop_warning = False
last_email_time = 0
last_image_path = None
clear_start_time = None
in_pir_mode = True  # start in PIR mode

def show_marquee_warning():
    while not stop_warning:
        show_message(device, "WARNING", fill="white", font=proportional(LCD_FONT), scroll_delay=0.05)

try:
    while True:
        # Wait for PIR if in idle mode
        if in_pir_mode:
            show_message(device, "IDLE", fill="white", font=proportional(LCD_FONT), scroll_delay=0.05)
            print("Waiting for motion...")
            pir.wait_for_motion()
            print("Motion detected! Starting YOLO detection loop...")
            in_pir_mode = False

        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(source=frame, save=False, show=False, conf=0.5, imgsz=224)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv11 Detection", annotated_frame)

        # Person detection and counting
        person_ids = [cls for cls in results[0].boxes.cls.cpu().numpy()] if results[0].boxes else []
        person_count = sum(1 for cls in person_ids if cls == 0)
        person_detected = person_count > 0

        print(f"Person count: {person_count}")

        if person_detected:
            no_person_start_time = None
            clear_start_time = None
            if detection_start_time is None:
                detection_start_time = time.time()
            elif time.time() - detection_start_time >= DETECTION_DURATION:
                if not warning_active:
                    warning_active = True
                    stop_warning = False
                    warning_thread = Thread(target=show_marquee_warning)
                    warning_thread.start()

                current_time = time.time()
                if current_time - last_email_time >= 10:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    image_path = f"snapshots/detected_{timestamp}.jpg"
                    cv2.imwrite(image_path, annotated_frame)
                    last_image_path = image_path
                    send_email_alert(image_path)
                    last_email_time = current_time

        else:
            detection_start_time = None
            if warning_active:
                if no_person_start_time is None:
                    no_person_start_time = time.time()
                elif time.time() - no_person_start_time >= WARNING_CLEAR_DELAY:
                    stop_warning = True
                    warning_thread.join()
                    warning_active = False
                    show_message(device, "CLEAR", fill="white", font=proportional(LCD_FONT), scroll_delay=0.05)
                    last_email_time = 0

                    # Wait and send snapshot for clear email
                    time.sleep(1)
                    ret, clear_frame = cap.read()
                    if ret:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        clear_image_path = f"snapshots/clear_{timestamp}.jpg"
                        cv2.imwrite(clear_image_path, clear_frame)
                        send_clear_email(clear_image_path)

                    clear_start_time = time.time()

        # Return to PIR mode if clear lasted long enough
        if not warning_active and clear_start_time:
            if time.time() - clear_start_time >= 10 and not in_pir_mode:
                print("Returned to idle PIR mode.")
                in_pir_mode = True
                clear_start_time = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    stop_warning = True
    if 'warning_thread' in locals():
        warning_thread.join()
    cap.release()
    cv2.destroyAllWindows()
