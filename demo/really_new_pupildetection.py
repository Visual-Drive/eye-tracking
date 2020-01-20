import cv2
import numpy as np
from math import hypot

eye_cascade = cv2.CascadeClassifier('../res/eye.xml')
cap = cv2.VideoCapture(0)


def detect_eyes(img):
    if img is not None:
        eyes = eye_cascade.detectMultiScale(img, 1.3, 5)
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

        if right_eye is not None:
            return right_eye
    else:
        return None


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]
    if img is None:
        print("Kein Auge erkennbar")
    else:
        return img


while True:
    ret, frame = cap.read()

    if ret is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye = detect_eyes(frame)

        if eye is not None:

            eye = cut_eyebrows(eye)

            # Find darkest part of picture
            min_max = cv2.minMaxLoc(eye)
            cv2.circle(eye, min_max[2], 1, 255, 3)

            # Calculate distance to left and right border of picture
            distance_left = hypot((min_max[2][0] - 0), (min_max[2][1] - eye.shape[0] / 2))
            distance_right = hypot((eye.shape[1] - min_max[2][0]), (eye.shape[0] / 2 - min_max[2][1]))
            # Draw corresponding lines
            cv2.line(eye, min_max[2], (0, min_max[2][1]), (255, 0, 0), 1)
            cv2.line(eye, min_max[2], (eye.shape[1], min_max[2][1]), (255, 0, 0), 1)

            # Direction detection
            if distance_right < eye.shape[1] / 3:
                cv2.putText(frame, 'LEFT', (50, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0))
                print('LEFT')
                # Calculate percentage and map to degrees
                left_percent = (distance_right / (eye.shape[1] / 2)) * 100
                degrees = 90 * (left_percent / 100)
                print(distance_right)
            if distance_left < eye.shape[1] / 2:
                cv2.putText(frame, 'RIGHT', (50, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0))
                print('RIGHT')
                # Calculate percentage and map to degrees
                right_percent = (distance_right / (eye.shape[1] / 2)) * 100
                degrees = 90 * (right_percent / 100)
                print(distance_left)

            cv2.imshow('eye', eye)

        cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
