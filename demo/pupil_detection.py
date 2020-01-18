import cv2
import numpy as np

print("abc")
print(cv2.__version__)

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

fehler = cv2.imread("../res/fehler.png")

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
        return False
    else:
        return left_eye

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]
    if img is None:
        print("Kein Auge erkennbar")
        return False
    else:
        return img


def blob_process(img):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    _, imgt = cv2.threshold(gray_frame, 22, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(_)
    imgt = cv2.erode(imgt, None, iterations=2)
    imgt = cv2.dilate(imgt, None, iterations=4)
    imgt = cv2.medianBlur(imgt, 5)
    _, contours, _ = cv2.findContours(imgt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    rows, cols, _ = img.shape
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 1)
        cv2.line(img, (x + int(w/2), 0), (x+int(w/2), rows), (0, 255, 0), 1)
        cv2.line(img, (0, y + int(h/2)), (cols, y + int(h/2)), (0,255,0), 1)
        center = (x + int(w/2), y + int(h/2))
        #print("{}, {}".format(x, int(cols/3)))
        cv2.circle(img, center, 5, (0, 0, 255), 1)
        get_direction(center[0], cols)
        break

    cv2.imwrite('pupil.png', img)
    cv2.imshow('my image', img)
    cv2.imshow('imgt', imgt)
    return img
    
def get_direction(x, cols):
    global state
    if x > int(cols/3) and x < int(cols/3) * 2 and state is not "middle":
        #TODO: Call communicatior.py
        state = "middle"
        print("middle")
    elif x < int(cols/3) and state is not "right":
        #TODO: Call communicatior.py
        state = "right"
        print("right")
    elif x > int(cols/3) * 2 and state is not "left":
        #TODO: Call communicatior.py
        state = "left"
        print("left")


cap = cv2.VideoCapture(0)
state = "middle"

while True:
    ret, frame = cap.read()
    frame = detect_eyes(frame, eye_cascade)
    if(frame is False):
        frame = fehler
    else:
        frame = cut_eyebrows(frame)
        keypoints = blob_process(frame)
        frame = cv2.resize(frame, (0,0), fx=2, fy=2, interpolation = cv2.INTER_LINEAR) 
    cv2.imshow("VisualDrive",frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()



