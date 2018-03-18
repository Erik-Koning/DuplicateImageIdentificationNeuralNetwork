#!/usr/bin/env python

#CISC 452
#Erik Koning
#March 14, 2018
#Python 2.7 built with cv2
#please set line 17

import cv2
import numpy as np
import os
import errno #EEXIST
import random #Uniform

from download_files import download_files

# 80% of data will be for training, 20% of data will be for testing/validation.
TRAIN_TEST_RATIO = 0.8
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

	"""
	print "Current working directory (should be: the 1 directory above where all the photo folders are) is \n"
	directory = os.getcwd()
	print directory

	###################
	#You must set this#
	###################
	pathToMyDesktop = "/home/ian/Desktop"

	"""
	
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
	categories = [dir for dir in os.listdir(filedir) if os.isdir(dir)]


	traincount = 0.01
	testcount = 0.01

	for category in categories:
		for imagepath in os.listdir(category):
			image = cv2.imread(imagepath)
			
	                # Image dimensions
        	        height, width = img.shape[:2]

			# Resize the image by a random ratio
			ratio = random.uniform(0.5,2.0)
			resized_image = cv2.resize(image, (height*ratio, width*ratio))

			# Now make both images 128x128
			image = cv2.resize(image, (HEIGHT, WIDTH))
			resized_image = cv2.resize(resized_image, (HEIGHT, WIDTH))

			

"""
	pathToRootDirectory = "ReSizedObjectCatagories"
	if not os.path.exists(pathToRootDirectory):
					os.makedirs(pathToRootDirectory)

    i = 0
    for root, dirs, files in os.walk(directory):

        #save each image directory name
        if(i == 1):
            folders = dirs

        #loop counter
        i+=1
        #print i,"\nroot:\n", root,"\ndirs:\n", dirs,"\nfiles:\n",files     #testing
        
        #look at each possible image in directory
        for file in files:
            print file
            if file.lower().endswith('.jpg'):
         
                # Read image
                img = cv2.imread(root+'\\'+file)		#image location, how the image should be read
                
                #image dimensions
                height, width = img.shape[:2]

                if(width > 128):
                    #scale down while keeping content of image, results in some stretching
                    ratio = width/128
                    height = height/ratio
                
                #resizing image
                resized_image = cv2.resize(img, (128, height))    #w , h
                
                #Accessor index for directory name in array "folders"
                imageIndex = file.split("_")[0]

                #file representative to image type
                path = pathToRootDirectory + '\\' + folders[int(imageIndex)-1]
                
                #make folder for resized file type if non existant
                if not os.path.exists(path):
                    os.makedirs(path)

                #write image to appropriate path with same file name
                cv2.imwrite(path+'\\'+file, resized_image)
                
                #cv2.imshow("scaled image",resized_image)
                #cv2.waitKey(0)

    path = "C:\Users\erik\Desktop\ReSizedObjectCatagories\Training"
    if not os.path.exists(path):
        os.makedirs(path)    
    path = "C:\Users\erik\Desktop\ReSizedObjectCatagories\Testing"
    if not os.path.exists(path):
        os.makedirs(path)    
    path = "C:\Users\erik\Desktop\ReSizedObjectCatagories\Validation"
    if not os.path.exists(path):
        os.makedirs(path)    
    print i
    cv2.waitKey(0)

"""


if __name__ == '__main__' :
	main()

