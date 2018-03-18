import urllib2
import os
import tarfile

# If the files are not already downloaded, do so.
def download_files():
	url = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"
	filename = "256_ObjectCategories.tar"
	
	# If we're not on windows, just use wget since it has a nice progress bar
	if os.name != 'nt':
		os.system("wget " + url)
	else:
		# No progress bar :(
		urllib.urlretrieve(url, filename)

	print("Finished download. Now extracting...")

	tar = tarfile.open(filename)
	tar.extractall()
	tar.close()

	print ("Extraction finished.")

	# Now that the tar archive is extracted, remove it.
	os.remove(filename)

if __name__ == "__main__":
	download_files()
