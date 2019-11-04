import cv2
from helper import detect_eyes, cut_eyebrows, blob_process

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = detect_eyes(frame, eye_cascade)
    frame = cut_eyebrows(frame)
    keypoints = blob_process(frame, detector)
    # cv2.drawKeypoints(frame, keypoints, frame, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    frame = cv2.resize(frame, (0,0), fx=2, fy=2) 
    cv2.imshow("my frame",frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()

