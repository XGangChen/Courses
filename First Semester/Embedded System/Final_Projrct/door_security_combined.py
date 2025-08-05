import cv2
import time

# import RPi.GPIO as GPIO
from gpiozero import MotionSensor

from ultralytics import YOLO
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.legacy import show_message
from luma.core.legacy.font import proportional, LCD_FONT

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
PIR_PIN = 17  # GPIO pin connected to HC-SR501 OUT
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(PIR_PIN, GPIO.IN)

pir = MotionSensor(17)
print("Waiting for motion...")
pir.wait_for_motion()
print("Motion detected!")

# --- Person detection tracking ---
DETECTION_DURATION = 10  # seconds
detection_start_time = None
warning_displayed = False

try:
    pir.wait_for_motion()
    print("Motion detected! Starting YOLO detection loop...")

    yolo_active = True

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model.predict(source=frame, save=False, show=False, conf=0.5, imgsz=224)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv11 Detection", annotated_frame)

        if results[0].boxes:
            if detection_start_time is None:
                detection_start_time = time.time()
            elif time.time() - detection_start_time >= DETECTION_DURATION and not warning_displayed:
                show_message(device, "WARNING", fill="white", font=proportional(LCD_FONT), scroll_delay=0.1)
                warning_displayed = True
        else:
            detection_start_time = None
            warning_displayed = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # while True:
    #     if pir.motion_detected:  # Motion detected
    #         ret, frame = cap.read()
    #         if not ret:
    #             continue

    #         # Without showing the frame
    #         # results = model.predict(source=frame, save=False, show=False, conf=0.5)   
            
    #         # Display the frame with detection results
    #         results = model.predict(source=frame, save=False, show=False, conf=0.5, imgsz=224)
    #         annotated_frame = results[0].plot()
    #         cv2.imshow("YOLOv11 Detection", annotated_frame)            

    #         if results[0].boxes:
    #             if detection_start_time is None:
    #                 detection_start_time = time.time()
    #             elif time.time() - detection_start_time >= DETECTION_DURATION and not warning_displayed:
    #                 # Show warning
    #                 show_message(device, "WARNING", fill="white", font=proportional(LCD_FONT), scroll_delay=0.1)
    #                 warning_displayed = True
    #         else:
    #             detection_start_time = None
    #             warning_displayed = False
    #     else:
    #         detection_start_time = None
    #         warning_displayed = False
    #         time.sleep(0.1)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
