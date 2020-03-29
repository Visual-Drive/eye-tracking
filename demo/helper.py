import cv2
import time
import numpy as np
from enum import Enum

blinked_count = 0
blinking_start = None
modes = Enum('Modes', 'DRIVE STOP')
current_mode = modes.STOP


def get_eye_pos(img, eyes):
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > img.shape[0] / 2:
            pass
        eyecenter = x + w / 2
        if eyecenter > img.shape[1] * 0.5:
            right_eye = img[y:y + h, x:x + w]
    if right_eye is not None:
        return right_eye


def detect_eyes2(img, eye_cascade_open, eye_cascade_open_or_closed):
    """
    Function to detect eyes
    returns True if the eye is closed, indicating a blink
    returns position of the right eye
    """
    if img is not None:
        eyes = eye_cascade_open.detectMultiScale(img, 1.3, 5)
        if len(eyes) > 0:
            right_eye = get_eye_pos(img, eyes)
            return False, right_eye
        else:
            return True, None
            '''
            eyes = eye_cascade_open_or_closed.detectMultiScale(img, 1.3, 5)
            if len(eyes) > 0:
                right_eye = get_eye_pos(img, eyes)
                return True, right_eye
            else:
                return None, None
                '''


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]
    if img is None:
        print("Kein Auge erkennbar")
    else:
        return img


def get_pupil_coords(eye):
    # Find darkest part of picture
    min_max = cv2.minMaxLoc(eye)
    # print(min_max)

    # Perform Eye-Center-localization with CDF approach
    pmi = min_max[2]
    # print(pmi)
    intensities = []
    try:
        # Scan 10x10 matrix for average intensity
        for x in range(pmi[0] - 10, pmi[0] + 10):
            for y in range(pmi[1] - 10, pmi[1] + 10):
                intensities.append(eye[y][x])
    except IndexError:
        pass

    # Calculate average intensity
    ai = sum(intensities) / len(intensities)
    # print(min_max[0], ai)

    # Create threshold with AI-filter
    _, thresh = cv2.threshold(eye, ai, 255, cv2.THRESH_BINARY_INV)

    # Opening to remove noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (2, 2))

    # Create 15x15 kernel
    region = [
        [pmi[0] - 14, pmi[1] - 14],
        [pmi[0] - 14, pmi[1] + 14],
        [pmi[0] + 14, pmi[1] + 14],
        [pmi[0] + 14, pmi[1] - 14]
    ]

    mask = np.array([region], np.int32)
    image2 = np.zeros((thresh.shape[0], thresh.shape[1]), np.uint8)
    cv2.fillPoly(image2, [mask], 255)

    # Filter threshold with kernel
    out = cv2.bitwise_and(thresh, thresh, mask=image2)

    cv2.imshow('thresh', thresh)
    cv2.imshow('out', out)

    # Calculate center of gravity
    M = cv2.moments(out)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY
