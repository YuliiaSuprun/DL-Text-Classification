# Task 3: Implementing a Feed-Forward Network.

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# the number of iterations to train for
numTrainingIters = 10000

# the number of hidden neurons that hold the state of the RNN
hiddenUnits = 1000

# the number of classes that we are learning over
numClasses = 3

# the number of data points in a batch
batchSize = 100


def addToData(maxSeqLen, data, testData, fileName, classNum, linesToUse, testLines):
    with open(fileName) as f:
        content = f.readlines()
    myInts = np.random.choice(
        range(len(content)), linesToUse + testLines, replace=False)
    i = len(data)
    test_i = len(testData)
    for whichLine in myInts.flat:
        line = content[whichLine]
        if line.isspace() or len(line) == 0:
            continue
        if len(line) > maxSeqLen:
            maxSeqLen = len(line)
        temp = np.zeros((len(line), 256))
        j = 0
        for ch in line:
            if ord(ch) >= 256:
                continue
            temp[j][ord(ch)] = 1
            j = j + 1
        if (testLines > 0):
            testData[test_i] = (classNum, temp)
            testLines -= 1
            test_i += 1
        else:
            data[i] = (classNum, temp)
            i = i + 1
    return (maxSeqLen, data, testData)


def pad(maxSeqLen, data):
    for i in data:
        temp = data[i][1]
        label = data[i][0]
        len = temp.shape[0]
        padding = np.zeros((maxSeqLen - len, 256))
        data[i] = (label, np.transpose(
            np.concatenate((padding, temp), axis=0)))
    return data


def generateDataFeedForward(maxSeqLen, data):
    myInts = np.random.random_integers(0, len(data) - 1, batchSize)
    x = np.stack(data[i][1].flatten() for i in myInts.flat)
    y = np.stack(np.array((data[i][0])) for i in myInts.flat)
    return (x, y)


# create the data dictionary
maxSeqLen = 0
data = {}
testData = {}
# load up the three data sets
(maxSeqLen, data, testData) = addToData(
    maxSeqLen, data, testData, "Holmes.txt", 0, 10000, 1000)
(maxSeqLen, data, testData) = addToData(
    maxSeqLen, data, testData, "war.txt", 1, 10000, 1000)
(maxSeqLen, data, testData) = addToData(
    maxSeqLen, data, testData, "william.txt", 2, 10000, 1000)

# pad each entry in the dictionary with empty characters as needed so
# that the sequences are all of the same length
data = pad(maxSeqLen, data)
testData = pad(maxSeqLen, testData)

# now we build the TensorFlow computation... there are two inputs,
# a batch of text lines and a batch of labels
inputX = tf.placeholder(tf.float32, [batchSize, 256 * maxSeqLen])
inputY = tf.placeholder(tf.int32, [batchSize])

# this is the inital state of the RNN, before processing any data
initialState = tf.placeholder(tf.float32, [batchSize, hiddenUnits])

# the weight matrix that maps the inputs and hidden state to a set of values
W = tf.Variable(np.random.normal(
    0, 0.05, (256 * maxSeqLen, hiddenUnits)), dtype=tf.float32)

# biases for the hidden values
b = tf.Variable(np.zeros((1, hiddenUnits)), dtype=tf.float32)

# weights and bias for the final classification
W2 = tf.Variable(np.random.normal(
    0, 0.05, (hiddenUnits, numClasses)), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, numClasses)), dtype=tf.float32)

# unpack the input sequences so that we have a series of matrices,
# each of which has a one-hot encoding of the current character from
# every input sequence
# sequenceOfLetters = tf.unstack(inputX, axis=1)

# now we implement timeTick = the forward pass

currentState = tf.tanh(tf.matmul(inputX, W) + b)
outputs = tf.matmul(currentState, W2) + b2

predictions = tf.nn.softmax(outputs)

# compute the loss
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=outputs, labels=inputY)
totalLoss = tf.reduce_mean(losses)

# use gradient descent to train
#trainingAlg = tf.train.GradientDescentOptimizer(0.02).minimize(totalLoss)
trainingAlg = tf.train.AdagradOptimizer(0.02).minimize(totalLoss)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

# and train!!
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #
    # initialize everything
    sess.run(tf.global_variables_initializer())
    #
    # and run the training iters
    for epoch in range(numTrainingIters):
        #
        # get some data
        x, y = generateDataFeedForward(maxSeqLen, data)
        #
        # do the training epoch
        _currentState = np.zeros((batchSize, hiddenUnits))
        _totalLoss, _trainingAlg, _currentState, _predictions, _outputs = sess.run(
            [totalLoss, trainingAlg, currentState, predictions, outputs],
            feed_dict={
                inputX: x,
                inputY: y,
                initialState: _currentState
            })
        # just FYI, compute the number of correct predictions
        numCorrect = 0
        for i in range(len(y)):
            maxPos = -1
            maxVal = 0.0
            for j in range(numClasses):
                if maxVal < _predictions[i][j]:
                    maxVal = _predictions[i][j]
                    maxPos = j
            if maxPos == y[i]:
                numCorrect = numCorrect + 1
        #
        # print out to the screen
        print("Step", epoch, "Loss", _totalLoss,
              "Correct", numCorrect, "out of", batchSize)
    # Save the session.
    saver = tf.train.Saver()
    saver.save(sess, 'checkPoint3.tf')
# Testing.

# Restore teh session.
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, 'checkPoint3.tf')

# Generate the testing dataset.
# Calculate total loss.
totalCorrect = 0
loss = 0

for k in range((int)(len(testData) / batchSize)):
    xTest = np.stack(testData[i][1].flatten()
                     for i in range(k * batchSize, (k+1) * batchSize))
    yTest = np.stack(np.array((testData[i][0]))
                     for i in range(k * batchSize, (k+1) * batchSize))
    _currentState = np.zeros((batchSize, hiddenUnits))
    _totalLoss, _predictions = sess.run([totalLoss, predictions], feed_dict={
        inputX: xTest, inputY: yTest, initialState: _currentState})
    loss += _totalLoss
    for i in range(len(yTest)):
        maxPos = -1
        maxVal = 0.0
        for j in range(numClasses):
            if maxVal < _predictions[i][j]:
                maxVal = _predictions[i][j]
                maxPos = j
        if maxPos == yTest[i]:
            totalCorrect += 1
print("Loss for", len(testData), "randomly chosen documents is", loss / (len(testData) /
      batchSize), ", number correct labels is", totalCorrect, "out of", len(testData))
