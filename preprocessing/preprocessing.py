#!/usr/bin/env python

#CISC 452
#Ian Chew, Erik Koning
#March 14, 2018
#Python 2.7 built with cv2

import cv2
import numpy as np
import os
import errno #EEXIST
import random #Uniform

# Local files, part of our project.
from download_files import download_files
import transformations

# 80% of data will be for training, 20% of data will be for testing/validation.
TRAIN_AMOUNT = 0.8
# Size of output images.
HEIGHT = 128
WIDTH = 128

# Creates the specified directory structure, if it doesn't exist already.
def make_path(path):
	try:
		os.makedirs(path)
	except OSError as e:
		# Ignore errors if the path already exists.
		if e.errno != errno.EEXIST:
			# Propagate any other errors.
			raise

def main():
	# Location of this script.
	path_to_script = os.path.dirname(os.path.realpath(__file__))

	# Change directory to the path containing this script so that we can call it from anywhere.
	os.chdir(path_to_script)

	# Path to ../data, where the data goes.
	datapath = os.path.join("..","data")
	
	make_path(datapath)

	# Directory to which the files will be downloaded.
	filedir = os.path.join(datapath, "256_ObjectCategories")
	# If we don't already have the files downloaded, download them.
	if not os.path.isdir(filedir):
		download_files(datapath)

	# Path to where the training and testing data will go.
	trainpath = os.path.join(datapath, "train")
	testpath = os.path.join(datapath, "test")

	make_path(trainpath)
	make_path(testpath)

	# Path to where the true and false cases will go.
	train_true_path = os.path.join(trainpath, "true_cases")
	train_false_path = os.path.join(trainpath, "false_cases")
	test_true_path = os.path.join(testpath, "true_cases")
	test_false_path = os.path.join(testpath, "false_cases")

	make_path(train_true_path)
	make_path(train_false_path)
	make_path(test_true_path)
	make_path(test_false_path)

	# 256_ObjectCategories is a directory of directories, each subdirectory
	# containing objects of one class.
	categories = (os.path.join(filedir,dir) for dir in os.listdir(filedir))
	categories = (path for path in categories if os.path.isdir(path)) #Filter out non-directories.

	traincount = 0
	testcount = 0
	total = 0.00001

	# The last image we scanned in.
	# On the first image, there is no previous image, so we'll
	# just use HEIGHTxWIDTH of black pixels to generate a wrong
	# image for false test cases.
	previous_image = np.zeros([HEIGHT, WIDTH, 3])

	for category in categories:
		imagefiles = (file for file in os.listdir(category) if os.path.isfile(os.path.join(category,file)))

		# Note: imagefile does not contain the whole path, just the name of the file.
		for imagefile in imagefiles:
			# The actual path to the image.
			imagepath = os.path.join(category, imagefile)
			print(imagepath)

			image = cv2.imread(imagepath)

			if image is None:
				# We read something that wasn't an image.
				continue #Just go on to the next file.

			# Resize the image by a random ratio
			resized_image = transformations.resize(image)

			# Now make both images 128x128
			image = cv2.resize(image, (HEIGHT, WIDTH))
			resized_image = cv2.resize(resized_image, (HEIGHT, WIDTH))

			# If we're above the ratio, we need more testing cases.
			if traincount/total > TRAIN_AMOUNT:
				testcount += 1
				# Directory to write the generated images.
				out_true_dir = test_true_path
				out_false_dir = test_false_path
			else:
				traincount += 1
				# Directory to write the generated images.
				out_true_dir = train_true_path
				out_false_dir = train_false_path

			# Split off the extension of the file and the filename itself.
			image_filename, image_ext = os.path.splitext(imagefile)

			# First image will have the same name as the original, but with "a"
			# on the end, second will have "b" on the end.
			im1dest = os.path.join(out_true_dir, image_filename+"a"+image_ext)
			im2dest = os.path.join(out_true_dir, image_filename+"b"+image_ext)
			
			# Write out the two images.
			cv2.imwrite(im1dest, image)
			cv2.imwrite(im2dest, resized_image)

			# Path to the pair of images for the false case.
			im1dest = os.path.join(out_false_dir, image_filename+"a"+image_ext)
			im2dest = os.path.join(out_false_dir, image_filename+"b"+image_ext)
			
			# Write out the two images.
			cv2.imwrite(im1dest, image)
			cv2.imwrite(im2dest, previous_image)
			
			previous_image = image
			total += 1


if __name__ == '__main__' :
	main()

