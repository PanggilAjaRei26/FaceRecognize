# pylint: disable=E0401

import cv2

CAMERA = 0
video = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
user_id = input ('Masukan ID : ')
A = 0
while True:
    A = A + 1
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)
    for x, y, w, h in wajah:
        cv2.imwrite('DataSet/User.'+str(user_id)+'.'+str(A)+'.jpg',abu[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Face Cam", frame)
    key = cv2.waitKey(1)
    if A > 29:
        break
video.release()
cv2.destroyAllWindows()
