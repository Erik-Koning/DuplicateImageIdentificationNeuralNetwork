# Dimensions of the image
HEIGHT=128
WIDTH=128

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

# Size of each batch, and the number of epochs to run.
BATCH_SIZE = 64
EPOCHS = 10

# Amount of seconds between status updates when importing data.
SECONDS_PER_PRINT = 1
