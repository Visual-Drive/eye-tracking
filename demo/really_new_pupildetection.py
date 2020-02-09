import cv2
import numpy as np
import time
from math import hypot

eye_cascade_open = cv2.CascadeClassifier('../res/eye.xml')
eye_cascade_open_or_closed = cv2.CascadeClassifier('../res/haarcascade_righteye_2splits.xml')
cap = cv2.VideoCapture(0)
blinked = 0
start = None


def detect_eyes(img):
    if img is not None:
        eyes = eye_cascade_open.detectMultiScale(img, 1.3, 5)
        global blinked
        if len(eyes) == 0:
            eyes = eye_cascade_open_or_closed.detectMultiScale(img, 1.3, 5)
            global start
            if len(eyes) == 0:
                return None
            if blinked == 0:
                start = time.time()
            blinked += 1
            if blinked == 3:
                end = time.time()
                elapsed = end - start
                print(elapsed)
                if elapsed <= 3:
                    print("Change modes")
                blinked = 0
            print(blinked)
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
            print(min_max)

            # Perform Eye-Center-localization with CDF approach
            pmi = min_max[2]
            print(pmi)
            intensities = []
            try:
                # Scan 10x10 matrix for average intensity
                for x in range(pmi[0] - 5, pmi[0] + 5):
                    for y in range(pmi[1] - 5, pmi[1] + 5):
                        intensities.append(eye[y][x])
            except IndexError:
                pass
            # Calculate average intensity
            ai = sum(intensities) / len(intensities)
            print(min_max[0], ai)
            # Create threshold with AI-filter
            _, thresh = cv2.threshold(eye, ai, 255, cv2.THRESH_BINARY_INV)
            # Opening to remove noise
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (2, 2))
            cv2.imshow('thresh', thresh)
            # Calculate center of gravity
            M = cv2.moments(thresh)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(eye, (cX, cY), 1, (255, 255, 255), 3)

            # Calculate distance to left and right border of picture
            distance_left = hypot((cX - 0), (cY - eye.shape[0] / 2))
            distance_right = hypot((cY - cX), (eye.shape[0] / 2 - cY))

            # Draw corresponding lines
            cv2.line(eye, (cX, cY), (0, cY), (255, 0, 0), 1)
            cv2.line(eye, (cX, cY), (eye.shape[1], cY), (255, 0, 0), 1)
            cv2.line(eye, (cX, cY), (cX, 0), (255, 0, 0), 1)
            cv2.line(eye, (cX, cY), (cX, eye.shape[0]), (255, 0, 0), 1)

            # Direction detection
            if distance_right < eye.shape[1] / 3:
                cv2.putText(frame, 'LEFT', (50, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0))
                print('LEFT')
                # Calculate percentage and map to degrees
                left_percent = (distance_right / (eye.shape[1] / 2)) * 100
                degrees = 90 * (left_percent / 100)
                print(distance_right)
            if distance_left < eye.shape[1] / 3:
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
