import cv2
import time
from gpiozero import MotionSensor
from ultralytics import YOLO
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.legacy import show_message
from luma.core.legacy.font import proportional, LCD_FONT
from threading import Thread

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
print("Waiting for motion...")
pir.wait_for_motion()
print("Motion detected! Starting YOLO detection loop...")

# --- Detection Tracking ---
DETECTION_DURATION = 10  # seconds to trigger warning
WARNING_CLEAR_DELAY = 3  # seconds of no detection to clear warning

detection_start_time = None
no_detection_start_time = None
warning_active = False
stop_warning = False

def show_marquee_warning():
    while not stop_warning:
        show_message(device, "WARNING", fill="white", font=proportional(LCD_FONT), scroll_delay=0.05)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(source=frame, save=False, show=False, conf=0.5, imgsz=224)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv11 Detection", annotated_frame)

        if results[0].boxes:
            no_detection_start_time = None  # reset countdown
            if detection_start_time is None:
                detection_start_time = time.time()
            elif time.time() - detection_start_time >= DETECTION_DURATION and not warning_active:
                warning_active = True
                stop_warning = False
                warning_thread = Thread(target=show_marquee_warning)
                warning_thread.start()
        else:
            detection_start_time = None
            if warning_active:
                if no_detection_start_time is None:
                    no_detection_start_time = time.time()
                elif time.time() - no_detection_start_time >= WARNING_CLEAR_DELAY:
                    stop_warning = True
                    warning_thread.join()
                    warning_active = False

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
