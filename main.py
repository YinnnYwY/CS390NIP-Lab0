import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784
NUM_CLASSES_iris = 3

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
ALGORITHM = "tf_net"


class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, activation, layerNum, neuronsPerLayer, learningRate=0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = activation
        self.layerNum = layerNum
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        if self.layerNum == 2:
            self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
            self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)
        elif self.layerNum == 3:
            self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
            self.W2 = np.random.randn(self.neuronsPerLayer, self.neuronsPerLayer)
            self.W3 = np.random.randn(self.neuronsPerLayer, self.outputSize)
        elif self.layerNum > 3:
            self.W = {}
            self.W[0] = np.random.randn(self.inputSize, self.neuronsPerLayer)
            for i in range(self.layerNum - 2):
                self.W[i + 1] = np.random.randn(self.neuronsPerLayer, self.neuronsPerLayer)
            self.W[self.layerNum - 1] = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __Act(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(x, 0)

    # Activation prime function.
    def __Derivative(self, x):
        if self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'relu':
            x[x <= 0] = 0
            x[x > 0] = 1
            return x

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i: i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs=50, minibatches=True, mbs=128):
        # TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        # Generate mini-batches:
        for i in range(epochs):
            xBatch = self.__batchGenerator(xVals, mbs)
            yBatch = self.__batchGenerator(yVals, mbs)
            for xBatch_j, yBatch_j in zip(xBatch, yBatch):
                if self.layerNum == 2:
                    layer1, layer2 = self.__forward(xBatch_j)
                    L2d = (yBatch_j - layer2) * self.__Derivative(layer2)
                    L1d = np.dot(L2d, self.W2.T) * self.__Derivative(layer1)
                    self.W1 += np.dot(xBatch_j.T, L1d) * self.lr
                    self.W2 += np.dot(layer1.T, L2d) * self.lr
                if self.layerNum == 3:
                    layer1, layer2, layer3 = self.__forward(xBatch_j)
                    L3d = (yBatch_j - layer3) * self.__Derivative(layer3)
                    L2d = np.dot(L3d, self.W3.T) * self.__Derivative(layer2)
                    L1d = np.dot(L2d, self.W2.T) * self.__Derivative(layer1)
                    self.W1 += np.dot(xBatch_j.T, L1d) * self.lr
                    self.W2 += np.dot(layer1.T, L2d) * self.lr
                    self.W3 += np.dot(layer2.T, L3d) * self.lr
                elif self.layerNum > 3:
                    delta = {}
                    layers = self.__forward(xBatch_j)
                    for i in reversed(range(self.layerNum)):
                        if i == self.layerNum - 1:
                            delta[i] = (yBatch_j - layers[i]) * self.__Derivative(layers[i])
                        else:
                            delta[i] = np.dot(delta[i + 1], self.W[i + 1].T) * self.__Derivative(layers[i])
                    for i in range(self.layerNum):
                        if i == 0:
                            self.W[i] += np.dot(xBatch_j.T, delta[i]) * self.lr
                        else:
                            self.W[i] += np.dot(layers[i - 1].T, delta[i]) * self.lr

    # Forward pass.
    def __forward(self, input):
        if self.layerNum == 2:
            layer1 = self.__Act(np.dot(input, self.W1))
            layer2 = self.__Act(np.dot(layer1, self.W2))
            return layer1, layer2
        if self.layerNum == 3:
            layer1 = self.__Act(np.dot(input, self.W1))
            layer2 = self.__Act(np.dot(layer1, self.W2))
            layer3 = self.__Act(np.dot(layer2, self.W3))
            return layer1, layer2, layer3
        elif self.layerNum > 3:
            layers = {}
            for i in range(self.layerNum):
                if i == 0:
                    layers[i] = self.__Act(np.dot(input, self.W[i]))
                else:
                    layers[i] = self.__Act(np.dot(layers[i - 1], self.W[i]))
            return layers

    # Predict.
    def predict(self, xVals):
        if self.layerNum == 2:
            _, layer2 = self.__forward(xVals)
            return layer2
        if self.layerNum == 3:
            _, layer2, layer3 = self.__forward(xVals)
            return layer3
        elif self.layerNum > 3:
            layers = self.__forward(xVals)
            return layers[self.layerNum - 1]


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


# =========================<Pipeline Functions>==================================

def getRawData(dataset='mnist'):
    if dataset == 'mnist':
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif dataset == 'iris':
        iris = load_iris()
        x = iris.data
        y = iris.target
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw, dataset='mnist'):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if dataset == 'mnist' or dataset == '2layer' or dataset == 'cnn':
        yTrainP = to_categorical(yTrain, NUM_CLASSES)
        yTestP = to_categorical(yTest, NUM_CLASSES)
        if dataset == 'cnn':
            xTrain = xTrain.reshape((xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1))
            xTest = xTest.reshape((xTest.shape[0], xTest.shape[1], xTest.shape[2], 1))
        else:
            # Flatten the raw data to 2D
            xTrain = xTrain.reshape(yTrainP.shape[0], IMAGE_SIZE)
            xTest = xTest.reshape(yTestP.shape[0], IMAGE_SIZE)
        # Add range reduction here (0-255 ==> 0.0-1.0)
        xTrain = xTrain / 255
        xTest = xTest / 255
    elif dataset == 'iris':
        yTrainP = to_categorical(yTrain, NUM_CLASSES_iris)
        yTestP = to_categorical(yTest, NUM_CLASSES_iris)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))


def trainModel(data, dataset='mnist', activation='sigmoid', layer=2):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        inputSize = IMAGE_SIZE if dataset == 'mnist' else xTrain.shape[1]
        model = NeuralNetwork_2Layer(inputSize=inputSize,
                                     outputSize=yTrain.shape[1],
                                     activation=activation,
                                     layerNum=layer,
                                     neuronsPerLayer=32)
        model.train(xTrain, yTrain)
        return model
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        if dataset == '2layer':
            model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
        elif dataset == 'cnn':
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
            print(model.summary())
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(xTrain, np.argmax(yTrain, axis=1), batch_size=128, epochs=15, verbose=1)
        return model
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        pred = model.predict(data)
        return pred
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        pred = model.predict(data)
        return pred
    else:
        raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    y_true = np.argmax(yTest, axis=1)
    y_pred = np.argmax(preds, axis=1)
    for i in range(y_pred.shape[0]):
        if np.array_equal(y_true[i], y_pred[i]):   acc = acc + 1
    accuracy = acc / y_pred.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("Classifier f1 Score: %f%%" % (f1_score(y_true, y_pred, average='micro') * 100))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print()


# =========================<Main>================================================

def main():
    if ALGORITHM == "custom_net":
        print("=========================<Custom neural net>==============================")
        # 2-layer, sigmoid, mnist
        dataset = 'mnist'
        activation = 'sigmoid'
        layer = 2
        print("dataset: %s" % dataset)
        print("activation: %s" % activation)
        print("layer number: %d\n" % layer)
        raw = getRawData(dataset)
        data = preprocessData(raw, dataset)
        model = trainModel(data[0], dataset, activation, layer)
        preds = runModel(data[1][0], model)
        evalResults(data[1], preds)

        print("\n\n=========================<EC>==============================")
        # EC: 3-layer, sigmoid, mnist
        dataset = 'mnist'
        activation = 'sigmoid'
        layer = 3
        print("dataset: %s" % dataset)
        print("activation: %s" % activation)
        print("layer number: %d\n" % layer)
        raw = getRawData(dataset)
        data = preprocessData(raw, dataset)
        model = trainModel(data[0], dataset, activation, layer)
        preds = runModel(data[1][0], model)
        evalResults(data[1], preds)

        # EC: n-layer, sigmoid, mnist
        dataset = 'mnist'
        activation = 'sigmoid'
        layer = 4
        print("dataset: %s" % dataset)
        print("activation: %s" % activation)
        print("layer number: %d\n" % layer)
        raw = getRawData(dataset)
        data = preprocessData(raw, dataset)
        model = trainModel(data[0], dataset, activation, layer)
        preds = runModel(data[1][0], model)
        evalResults(data[1], preds)

        # EC: 2-layer, relu, mnist
        dataset = 'mnist'
        activation = 'relu'
        layer = 2
        print("dataset: %s" % dataset)
        print("activation: %s" % activation)
        print("layer number: %d\n" % layer)
        raw = getRawData(dataset)
        data = preprocessData(raw, dataset)
        model = trainModel(data[0], dataset, activation, layer)
        preds = runModel(data[1][0], model)
        evalResults(data[1], preds)

        # EC: 2-layer, sigmoid, iris
        dataset = 'iris'
        activation = 'sigmoid'
        layer = 2
        print("dataset: %s" % dataset)
        print("activation: %s" % activation)
        print("layer number: %d\n" % layer)
        raw = getRawData(dataset)
        data = preprocessData(raw, dataset)
        model = trainModel(data[0], dataset, activation, layer)
        preds = runModel(data[1][0], model)
        evalResults(data[1], preds)

    if ALGORITHM == "tf_net":
        print("=========================<TF neural net>==============================")
        # 2-layer, sigmoid, mnist, >95%
        descripton = '2layer'
        print("TF 2-layer NN\n")
        raw = getRawData()
        data = preprocessData(raw, descripton)
        model = trainModel(data[0], descripton)
        preds = runModel(data[1][0], model)
        evalResults(data[1], preds)

        print("=========================<EC>==============================")
        # mnist, >99%
        descripton = 'cnn'
        print("TF CNN\n")
        raw = getRawData()
        data = preprocessData(raw, descripton)
        model = trainModel(data[0], descripton)
        preds = runModel(data[1][0], model)
        evalResults(data[1], preds)


if __name__ == '__main__':
    main()

