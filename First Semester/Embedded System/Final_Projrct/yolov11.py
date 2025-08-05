from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')  # ä½¿ç¨è¼éæ¨¡åï¼é¿å Pi å¡ä½

cap = cv2.VideoCapture(0)  # å¦ææå¤å€æå½±è£ç½®ï¼å¯è½é€æ¹çº 1 æ "/dev/video0"
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # æ¨è«ï¼å³æç«é¢ï¼
    results = model.predict(source=frame, save=False, show=True, conf=0.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()