import cv2
import numpy as np

print("abc")
print(cv2.__version__)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


#img = cv2.imread("augen.jpg")
fehler = cv2.imread("fehler.png")
fehlermeldung = False
#gray_picture2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#faces = face_cascade.detectMultiScale(gray_picture2, 1.3, 5)


detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

def detect_eyes(img, classifier):
    if (img is not None):
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
        else:
            return left_eye
    else:
        return None


def detect_faces(img, classifier):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = classifier.detectMultiScale(gray_frame, 1.3, 5)
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

def cut_eyebrows(img):
    if (img is not None):
        height, width = img.shape[:2]
        eyebrow_h = int(height / 4)
        img = img[eyebrow_h:height, 0:width]
        if img is None:
            print("Kein Auge erkennbar")
        else:
            return img
    else:
        return None


def blob_process(img, detector):
    if (img is not None):
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, imgt = cv2.threshold(gray_frame, 60, 255, cv2.THRESH_BINARY_INV)
        # img = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1) #Funktioniert noch nicht

        imgt = cv2.erode(imgt, None, iterations=2)
        imgt = cv2.dilate(imgt, None, iterations=4)
        imgt = cv2.medianBlur(imgt, 5)
        contours, _ = cv2.findContours(imgt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

        cv2.imshow('my image', img)
        cv2.imshow('imgt', imgt)
        return img
    else:
        return None
        #keypoints = detector.detect(img)
        #print(img.shape)
        #for keyPoint in keypoints:
        #print(keyPoint.pt[0])
        #print(keyPoint.pt[1])
        #kreis_mittelpunkt = (int(keyPoint.pt[0]), int(keyPoint.pt[1]))
        #cv2.circle(img,kreis_mittelpunkt, 5, (255,0,0), 2  )
        #if keyPoint.pt[1] > 16.5:
            #print("oben")
    #print(keypoints[0].pt[0])

    #if keypoints is None:
        #print("Kein Auge erkennbar")
        #return fehler
    #else:
        #return keypoints


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = detect_eyes(frame, eye_cascade)
    frame = cut_eyebrows(frame)
    blob_process(frame, detector)
    if blob_process(frame, detector) is None:
        print("NONE")
        cv2.imshow('fehler', fehler)
    if blob_process(frame, detector) is not None:
        cv2.destroyWindow('fehler')
        print("NICHT NONE")
    key = cv2.waitKey(30)
    if key == 27:
        break


#cv2.imshow('my image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



