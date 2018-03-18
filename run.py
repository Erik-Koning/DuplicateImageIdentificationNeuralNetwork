#!/usr/bin/env python

#Runs the preprocessor, then the training program.
from __future__ import print_function

from subprocess import Popen #Create subprocesses
import os

def main():
	pppath = os.path.join("preprocessing", "preprocessing.py")
	trainpath = os.path.join("training", "train.py")

	print(pppath)

	print("Running preprocessing script...")
	Popen(["python" , pppath]).wait()
	Popen("ls").wait()

	print("Running training script...")
	Popen(["python", trainpath]).wait()

if __name__ == "__main__":
	main()
