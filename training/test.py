#!/usr/bin/env python

# Only test, don't train.

import train
import config

config.DO_TRAIN = False
config.DO_TEST = True

if __name__ == "__main__":
	import sys
	train.main(sys.argv)
