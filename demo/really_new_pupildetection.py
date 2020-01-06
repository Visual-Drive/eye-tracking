import cv2
from math import hypot

eye_cascade = cv2.CascadeClassifier('eye.xml')
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
                if y > frame.shape[0] / 2:
                    pass
                eye = frame[y:y + h, x:x + w]
                eye = cut_eyebrows(eye)
                # eye = cv2.Canny(eye, 100, 20)
                # _, eye = cv2.threshold(eye, 107, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                min_max = cv2.minMaxLoc(eye)
                _, thresh = cv2.threshold(eye, min_max[0] + 10, 255, cv2.THRESH_BINARY)
                cv2.imshow('thresh', thresh)
                im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                maxArea = 0

                for contour in contours:
                    area = cv2.contourArea(contour)
                    rect = cv2.boundingRect(contour)
                    '''
                    squareCoeff = rect[1] / rect[0]
                    print(squareCoeff)

                    SQUARE_COEFF = 1.5

                    if area > maxArea and SQUARE_COEFF > squareCoeff > 1.0 / SQUARE_COEFF:
                        maxArea = area
                        maxContourRect = rect

                try:
                    print(maxContourRect)
                except NameError:
                    print("error")
                if maxArea == 0:
                    print("Not found")
                # else:
                    #cv2.rectangle(eye, (maxContourRect.))
                    '''

                cv2.circle(eye, min_max[2], 1, 255, 3)
                '''
                            if min_max[2][0] > 2 * (eye.shape[1]/3):
                    # print(min_max[2][0])
                    # print(eye.shape[1])
                    print("left")
                elif min_max[2][0] < (eye.shape[1]/2):
                    print("right")
                '''

                # canny = cv2.Canny(eye, 60, 180)
                # cv2.imshow('canny', canny)

                distance_left = hypot((min_max[2][0] - 0), (min_max[2][1] - eye.shape[0] / 2))
                print(distance_left)
                cv2.line(eye, min_max[2], (eye.shape[1], int(eye.shape[0] / 2)), (255, 0, 0), 1)

                if distance_left > 30:
                    cv2.putText(frame, 'LEFT', (50, 150), cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 0))

                cv2.imshow('eye', eye)
                cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
