import urllib2
import os
import tarfile

# If the files are not already downloaded, do so.
# Path is the location to put the files.
def download_files(path):
	url = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"
	filename = "256_ObjectCategories.tar"
	
	filepath = os.path.join(path, filename)

	# If we're not on windows, just use wget since it has a nice progress bar
	if os.name != 'nt':
		os.system("wget " + url + " -O " + filepath)
	else:
		# No progress bar :(
		urllib2.urlretrieve(url, filepath)

	print("Finished download. Now extracting...")

	tar = tarfile.open(filepath)
	tar.extractall(path)
	tar.close()

	print ("Extraction finished.")

	# Now that the tar archive is extracted, remove it.
	os.remove(filepath)

if __name__ == "__main__":
	download_files()
