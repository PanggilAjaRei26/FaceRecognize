

# pylint: disable=E1101
# pylint: disable=E0401

import cv2

CAMERA = 0
video = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
faceDeteksi = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create() # type: ignore
recognizer.read('DataSet/training.yml')
A = 0

while True:
    A = A + 1
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)

    for x, y, w, h in wajah:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        USER_ID, conf = recognizer.predict(abu[y:y+h, x:x+w])
        if USER_ID == 1:
            USER_ID = 'Rei'
        elif USER_ID == 2:
            USER_ID = 'Alfin'
        elif USER_ID == 3:
            USER_ID = 'Hamzah'
        elif USER_ID == 4:
            USER_ID = 'Yuli'
        cv2.putText(frame, str(USER_ID), (x + 40, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))

    cv2.imshow("Face Cam", frame)
    key = cv2.waitKey(1)

    if key == ord("a"):
        break

video.release()
cv2.destroyAllWindows()
