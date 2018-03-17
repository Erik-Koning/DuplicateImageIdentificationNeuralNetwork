#CISC 452
#Erik Koning
#March 14, 2018
#Python 2.7 built with cv2
#please set line 17

import cv2
import numpy as np
import os

if __name__ == '__main__' :
	print "Current working directory (should be: the 1 directory above where all the photo folders are) is \n"
	directory = os.getcwd()
	print directory

	###################
	#You must set this#
	###################
	pathToMyDesktop = "C:\Users\erik\Desktop"


	pathToRootDirectory = pathToMyDesktop + "\ReSizedObjectCatagories"
	if not os.path.exists(pathToRootDirectory):
					os.makedirs(pathToRootDirectory)

	i = 0
	for root, dirs, files in os.walk(directory):

		#save each image directory name
		if(i == 1):
			folders = dirs

		#loop counter
		i+=1
		#print i,"\nroot:\n", root,"\ndirs:\n", dirs,"\nfiles:\n",files		#testing
		
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
				resized_image = cv2.resize(img, (128, height))	  #w , h
				
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
			
	print i
	cv2.waitKey(0)
