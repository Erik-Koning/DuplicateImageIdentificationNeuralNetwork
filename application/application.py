import Tkinter as tk
import tkFileDialog
from PIL import ImageTk, Image
import os

# Get helper functions
from helper import *

# Asks the user for a file, and puts that image in a label.
# Returns the opened image because we need to keep that from getting garbage collected.
def get_file(path, label):
	image = open_image(path)
	label.config(image=image)
#	label.pack()
	return image

def open_image(path):
	return ImageTk.PhotoImage(Image.open(path))

# Generates every pair of elements from a list. O(n^2)
def get_pairs(lis):
	for i,left in enumerate(lis):
		for j in range(i+1,len(lis)):
			right = lis[j]
			yield left,right

# Read in two images, and return the confidance that the two are duplicates.
def get_confidance(left, right):
	return 0.9

def main(argv):

	# Called when we skip the image.
	def skip():
		while True:
			# Get the next two images.
			try:
				left,right = next(filepairs)
			except StopIteration as e:
				# The program is over.
				root.destroy()
			# Get the confidance that the images are duplicates
			confidance = get_confidance(left, right)
			if confidance < 0.6:
				# Try next pair.
				continue


			images[0] = get_file(left, imlabel1)
			images[1] = get_file(right, imlabel2)
			msg = "Which one to keep? " + str(confidance*100) +\
			      "% confidant that the images are duplicates."
			toplabel.config(text=msg)
			print(left)
			print(right)
			break

	# Initialize the Tk root widget.
	root = tk.Tk()
	root.title("Image deduplicator")

	# Create the label widget, as a child of the root widget.
	toplabel = tk.Label(root, fg="blue", text="Which one to keep?")

	# Pack allows the window to fit the size of the text.
	toplabel.pack()

	# Will contain the two images.
	middle_label = tk.Label(root)
	middle_label.pack()

	imlabel1 = tk.Label(middle_label)
	imlabel1.pack(side="left")
	imlabel2 = tk.Label(middle_label)
	imlabel2.pack(side="right")

	images = []

	images.append(get_file("jpeg-home.jpg", imlabel1))
	images.append(get_file("compressed.jpg", imlabel2))

	buttons = tk.Label(root)
	left_button = tk.Button(buttons, text="Keep left", width=25, command=skip)
	left_button.grid(column=0, row=0)
	skip_button = tk.Button(buttons, text="Skip", width=25, command=skip)
	skip_button.grid(column=1, row=0)
	right_button = tk.Button(buttons, text="Keep right", width=25, command=skip)
	right_button.grid(column=2, row=0)

	buttons.pack()
#	button = tk.Button(root, text='Open', width=25, command = get_file)
#	button.pack()

	# Get a directory from the user to search for images in.
	directory = tkFileDialog.askdirectory()

	filenames = searchdir(directory, "*.jpg")

	print(filenames)

	filepairs = get_pairs(filenames)

	skip()

	# Do the main GUI event loop.
	root.mainloop()


import sys

if __name__ == "__main__":
	main(sys.argv)
