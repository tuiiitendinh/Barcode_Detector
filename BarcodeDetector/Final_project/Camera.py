import cv2
from BarcodeDetection_Camera import AreaDetection
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)
cap.set(3,637)
cap.set(4,280)
name_code = ""
while True:
    ret, frame = cap.read()
    b,r = AreaDetection(frame)
    if r is not None: cv2.imshow("R",r)
    cv2.imshow("B",b)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyWindow()