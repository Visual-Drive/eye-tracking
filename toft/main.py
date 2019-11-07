import cv2
import numpy as np

print('Currently running on: ' + cv2.__version__)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

fehler = cv2.imread("../res/fehler.png")

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame

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
    if (img is not None):
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, imgt = cv2.threshold(gray_frame, 25, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite('thresholded_image.png', imgt)
        # img = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1) #Funktioniert noch nicht

        imgt = cv2.erode(imgt, None, iterations=2)
        imgt = cv2.dilate(imgt, None, iterations=4)
        imgt = cv2.medianBlur(imgt, 5)
        _, contours, _ = cv2.findContours(imgt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        print(contours)
        rows, cols, _ = img.shape
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            #cv2.drawContours(img, [cnt], -1, (0,0,255), 3)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 1)
            cv2.line(img, (x + int(w/2), 0), (x+int(w/2), rows), (0, 255, 0), 1)
            cv2.line(img, (0, y + int(h/2)), (cols, y + int(h/2)), (0,255,0), 1)
            break

        cv2.imwrite('pupil.png', img)
        cv2.imshow('my image', img)
        cv2.imshow('imgt', imgt)
        return img
    else:
        return None


cap = cv2.VideoCapture(0)
#cap.set(3, 1280)
#cap.set(4, 720)

while True:
    ret, frame = cap.read()
    '''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    '''
    #face_frame = detect_faces(frame, face_cascade)
    #if face_frame is not None:
    eyes = detect_eyes(frame, eye_cascade)
    for eye in eyes:
        if eye is not None:
            eye = cut_eyebrows(eye)
            keypoints = blob_process(eye, detector)
            #eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
   


cap.release()
cv2.destroyAllWindows()



