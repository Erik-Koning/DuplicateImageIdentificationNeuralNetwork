import os
import fnmatch

# Return paths to everything in the directory that matches the glob.
def searchdir(path, glob):
	# List of paths to return
	out = []
	for root, _, filenames in os.walk(path):
		for filename in fnmatch.filter(filenames, glob):
			out.append(os.path.join(root, filename))
	return out
