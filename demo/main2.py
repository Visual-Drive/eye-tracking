import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fehler = cv2.imread("fehler.png")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_eyes(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_frame, 1.3, 5)
    width = np.size(img, 1)
    height = np.size(img, 0)
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]


    if left_eye is None:
        print("Kein Auge erkennbar")
        return fehler
    else:
        return left_eye

while True:
    ret, frame = cap.read()
    if ret is True:
        cv2.imshow('frame', frame)
        grey = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 1)
        _, threshold = cv2.threshold(grey, 3, 255, cv2.THRESH_BINARY_INV)
        eye = detect_eyes(frame, eye_cascade)
        cv2.imshow('eyes', eye)
        cv2.imshow('grey', grey)
        cv2.imshow("Threshold", threshold)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
