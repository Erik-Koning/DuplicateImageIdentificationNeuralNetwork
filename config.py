# Dimensions of the image
HEIGHT=128
WIDTH=128

# Amount of the data used for training (.8 = 80%)
TRAIN_AMOUNT = 0.8

# Number of convolution/max-pool layers
LAYERS = 2

# Number of channels after each layer
CHANNELS = [32, 64]

# Size of the kernel in the convolution layers.
KERNEL_SIZE = 5

# Size of the filter used in the max pooling layers.
POOL_SIZE = 2

# Number of nodes in the second-to-last fully connected layer
DENSE_NODES = 1024

# Learning rate
LEARNING_RATE = 0.001

# Size of each batch, and the number of epochs to run.
BATCH_SIZE = 64
EPOCHS = 5

# Amount of seconds between status updates when importing data.
SECONDS_PER_PRINT = 1

# How much of the data to import
DATA_FRACTION = 1

# If true, do training
DO_TRAIN = True

# If true, do testing
DO_TEST = False
