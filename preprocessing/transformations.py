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

# Compress an image using JPEG in a very lossy way.
# Quality is a number from 0 to 100, with higher numbers representing
# higher quality.
def compress(image, quality=10):
	# Flag list to send to imencode.
	flags = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
	res, image = cv2.imencode('.jpg', image, flags)
	return cv2.imdecode(image, 1)

# Crop the image by a random amount.
# lbound=0.5 => crop at most half the image off on each axis.
def crop(image, lbound = 0.5, hbound=1.0):
	# Need to divide the amount to delete by 2, since
	# we will delete off of both ends.
	lbound = 1 - (1 - lbound)/2
	height, width = image.shape[:2]
	# New lower bound of the height
	hlow = int((1-random.uniform(lbound, hbound))*height)
	# New upper bound of the height (will be cropped from hlow:hhigh)
	hhigh = int(random.uniform(lbound, hbound)*height)
	wlow = int((1-random.uniform(lbound, hbound))*width)
	whigh = int(random.uniform(lbound, hbound)*width)

	# Do a slice to crop the image.
	return image[hlow:hhigh, wlow:whigh]
