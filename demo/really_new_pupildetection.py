import cv2
import time
import helper
from enum import Enum

eye_cascade_open = cv2.CascadeClassifier('../res/eye.xml')
eye_cascade_open_or_closed = cv2.CascadeClassifier('../res/haarcascade_righteye_2splits.xml')
cap = cv2.VideoCapture(0)
# ser = serial.Serial('COM4', baudrate=19200, bytesize=serial.EIGHTBITS, stopbits=serial.STOPBITS_ONE)

is_gazing = False
blinked_count = 0
blinking_start = None

mode_start = None
modes = Enum('Modes', 'DRIVE STOP')
current_mode = modes.STOP

previously_closed = False

while True:
    ret, frame = cap.read()

    if ret is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = frame[0: frame.shape[0],int(frame.shape[1]/16):frame.shape[1]]
        # eye = helper.detect_eyes(frame, eye_cascade_open, eye_cascade_open_or_closed)
        is_closed, eye = helper.detect_eyes2(frame, eye_cascade_open, eye_cascade_open_or_closed)

        if is_closed and not previously_closed:
            if blinked_count == 0: blinking_start = time.time()
            blinked_count += 1
            if blinked_count >= 3:
                if time.time() - blinking_start <= 3:
                    current_mode = modes.DRIVE if current_mode is modes.STOP else modes.STOP
                blinked_count = 0

        previously_closed = is_closed

        if eye is not None:

            eye = helper.cut_eyebrows(eye)
            # eye = cv2.medianBlur(eye, 3)

            if current_mode is modes.DRIVE:
                cX, cY = helper.get_pupil_coords(eye)
                cv2.circle(eye, (cX, cY), 1, (255, 255, 255), 3)

                # Calculate distance to left and right border of picture
                distance_left = eye.shape[1] - cX
                distance_right = cX
                distance_top = cY
                distance_bottom = eye.shape[0] - cY

                # Draw corresponding lines
                cv2.line(eye, (cX, cY), (0, cY), (255, 0, 0), 1)
                cv2.line(eye, (cX, cY), (eye.shape[1], cY), (255, 0, 0), 1)
                cv2.line(eye, (cX, cY), (cX, 0), (255, 0, 0), 1)
                cv2.line(eye, (cX, cY), (cX, eye.shape[0]), (255, 0, 0), 1)

                # Direction detection
                
                if distance_right < eye.shape[1] / 3:
                    print('right')
                    cv2.putText(frame, 'RIGHT', (50, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0))
                    # print('LEFT')
                    # Calculate percentage and map to degrees
                    left_percent = (distance_right / (eye.shape[1] / 2)) * 100
                    degrees = 90 * (left_percent / 100)
                    # print(distance_right)
                elif distance_left < eye.shape[1] / 3:
                    print('left')
                    cv2.putText(frame, 'LEFT', (50, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0))
                        # print('RIGHT')
                        # Calculate percentage and map to degrees
                    right_percent = (distance_right / (eye.shape[1] / 2)) * 100
                    degrees = 90 * (right_percent / 100)
                        # print(distance_left)
                else:
                    cv2.putText(frame, 'CENTER', (50, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0))
                    if not is_gazing:
                        is_gazing = False
                        helper.mode_start = None
                
                if distance_top < eye.shape[0] / 3:
                    cv2.putText(frame, 'TOP', (50, 250), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0))
                else:
                    cv2.putText(frame, 'BOTTOM', (50, 250), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0))
                # print('Distance Right: ', distance_right, ', Distance Left: ', distance_left)
                
                cv2.imshow('eye', eye)
        cv2.putText(frame, f'Blinked: {blinked_count}', (50, 350), cv2.FONT_HERSHEY_PLAIN, 6, 255)
        cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        # ser.close()
        break

cv2.destroyAllWindows()
cap.release()
