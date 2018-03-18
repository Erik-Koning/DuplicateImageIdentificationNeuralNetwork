from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Library imports
import numpy as np
import cv2 # cv2.imread for reading images.
import os # path
import time

# Local file imports
from config import * #Configuration options like HEGIHT or WIDTH

# Calls os.listdir, but returns the full path to each object.
def list_full_path(path):
	return [os.path.join(path,x) for x in os.listdir(path)]

# Combines the two imported images into one numpy.ndarray with 6 "channels"
def combine_images(im1, im2):
	combined = np.array([im1, im2])
	transposed = np.transpose(combined, (1,2,0,3))
	return transposed.reshape(HEIGHT, WIDTH, 6) # 6 channels

# Returns a tuple containing (data, labels), where data is a 4-d array and labels is a 1D array.
# kind should be train, test, or validation
def get_data(kind):
	# Path to the data
	testpath = os.path.join('..', 'data', kind)
	# Paths to the true and false test cases
	truepath = os.path.join(testpath, 'true_cases')
	falsepath = os.path.join(testpath, 'false_cases')

	print("Reading True cases...")

	# The data we will return.
	data = []
	labels = [] #The labels we will return.
	files = sorted(list_full_path(truepath))
	# The following will be used for console output.
	testcount = len(files)//2
	oldtime = time.time()
	imported = 0 # Number of imported cases so far.
	for f1,f2 in zip(files[::2], files[1::2]):
		# Go through pairs of files.
		im1 = cv2.imread(f1)
		im2 = cv2.imread(f2)
		
		data.append(combine_images(im1, im2))
		labels.append(True) #This is a true case.
		
		if time.time() - oldtime > SECONDS_PER_PRINT:
			oldtime = time.time()
			print ("Imported", imported, "of", testcount, "cases.")
		
		imported += 1

	print("Reading False cases...")

	# Now import the false test cases.
	files = sorted(list_full_path(falsepath))
	# The following will be used for console output.
	testcount = len(files)//2
	oldtime = time.time()
	imported = 0 # Number of imported cases so far.
	for f1,f2 in zip(files[::2], files[1::2]):
		# Go through pairs of files.
		im1 = cv2.imread(f1)
		im2 = cv2.imread(f2)
		
		data.append(combine_images(im1, im2))
		labels.append(False) #This is a false case.

		if time.time() - oldtime > SECONDS_PER_PRINT:
			oldtime = time.time()
			print ("Imported", imported, "of", testcount, "cases.")
		
		imported += 1

	return np.array(data), np.array(labels)

