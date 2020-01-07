import cv2
import numpy as np
from math import hypot

eye_cascade = cv2.CascadeClassifier('../res/eye.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


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
        eyes = eye_cascade.detectMultiScale(frame, 1.3, 5)
        if eyes is not None:
            for (x, y, w, h) in eyes:
                if y > np.size(frame, 0) / 2:
                    pass
                eye = frame[y:y + h, x:x + w]
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

                distance_ratio = distance_left / distance_right
                print(distance_ratio)

                if distance_ratio > 1.5:
                    if distance_right > distance_left:
                        cv2.putText(frame, 'RIGHT', (50, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0))
                    elif distance_left > distance_right:
                        cv2.putText(frame, 'LEFT', (50, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0))

                cv2.imshow('eye', eye)
                cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
