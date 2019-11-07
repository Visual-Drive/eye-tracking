import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_hist(img):
	#img = cv2.resize(img, (0, 0), fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)
	n, bins, patches = plt.hist(img.ravel(), 256)
	plt.show()
