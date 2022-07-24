# Text Classification with Deep Learning

In this project, I used TensorFlow to implement some deep learning architectures used to classify sequences of raw text.
Since the algorithms I implemented are computationally expensive, I used Amazon's Deep Learning machine (on EC2) for training,
which has some preinstalled DL tools.

Task1: implements a classic RNN that tries to determine what file each line of text came from,
by only looking at the sequence of characters. Evaluates the accuracy of the RNN on a test set.

Task2: implements an RNN with "time warping" that helps solve the problem of vanishing gradient. Rather than
the state of the network being fed-forward only to the next time tick, the state is also fed forward ten time ticks into the future.

Task3: implements a simple feed-forward network with one hidden layer.

Task4: added a convolutional filter to the "time warping" RNN.
