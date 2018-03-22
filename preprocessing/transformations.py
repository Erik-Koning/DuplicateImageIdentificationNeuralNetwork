# The following are transformations to be used to create training data.

import cv2
import numpy as np
import random #Uniform

# Resizes an image randomly
def resize(image, lbound = 0.5, hbound = 2.0):
	height, width = image.shape[:2]
	# Resize the image by a random ratio
	ratio = random.uniform(lbound, hbound)
	# Need to cast height and width to an integer.
	return cv2.resize(image, (int(height*ratio), int(width*ratio)))
