import cv2
import numpy as np
from matplotlib import pyplot as plt

print('Currently running on: ' + cv2.__version__)

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

fehler = cv2.imread("../res/fehler.png")

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
            '''
            plt.subplot(2, 1, 1), plt.imshow(left_eye, cmap = 'gray')
            plt.title('Original Noisy Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(2, 1, 2), plt.hist(left_eye.ravel(), 256)
            plt.title('Histogram'), plt.xticks([]), plt.yticks([])
            plt.show()
            '''
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]
    if img is None:
        print("Kein Auge erkennbar")
        return False
    else:
        return img


def blob_process(img, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, 13, 255, cv2.THRESH_BINARY)
    # img = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1) #Funktioniert noch nicht
    img = cv2.erode(img, None, iterations=2)
    # cv2.imshow('img', img)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    if keypoints is None:
        print("Kein Auge erkennbar")
        return False
    else:
        print(keypoints)
        return keypoints


cap = cv2.VideoCapture(0)
#cap.set(3, 1280)
#cap.set(4, 720)

while True:
    ret, frame = cap.read()
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    '''
    eyes = detect_eyes(frame, eye_cascade)
    for eye in eyes:
        if eye is not None:
            eye = cut_eyebrows(eye)
            keypoints = blob_process(eye, detector)
            eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    '''
    frame = detect_eyes(frame, eye_cascade)
    if(frame is False):
        frame = fehler
    else:
        frame = cut_eyebrows(frame)
        keypoints = blob_process(frame, detector)
        cv2.drawKeypoints(frame, keypoints, frame, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        frame = cv2.resize(frame, (0,0), fx=2, fy=2, interpolation = cv2.INTER_LINEAR) 
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # _, img = cv2.threshold(gray_frame, 120, 255, cv2.THRESH_BINARY)
        # contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMLPE)
        # contours = sorted(contours, key = lambda x: cv2.contourArea(x), reverse = True)
    cv2.imshow("VisualDrive",frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    '''


cap.release()
cv2.destroyAllWindows()



