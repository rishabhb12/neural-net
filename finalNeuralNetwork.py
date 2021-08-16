import numpy as np
import os, sys
import csv
from random import shuffle
import math
import matplotlib.pyplot as plt


class NeuralNetwork:
    
    def __init__(self, input_dim, output_dim, NNodes):
        
        #set weight and bias here to access in and out of code
        self.W1 = np.random.randn(input_dim, NNodes) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, NNodes))
        self.W2 = np.random.randn(NNodes, output_dim) / np.sqrt(NNodes)
        self.b2 = np.zeros((1, output_dim))


    def fit(self,X,Y, learningRate, epochs, regLambda):

        #fit using forward and back prop
        #use soft max since have 2 output nodes 
        for i in range(0, epochs):
            output1, output2 = self.forward(X)
            eoutput2 = np.exp(output2)
            softmax = eoutput2 / np.sum(eoutput2, axis=1, keepdims=True)
            
            self.backpropagate(X, Y, learningRate, regLambda, output1, softmax)

            
        return 0
            
    def forward(self, X):
        
        #matrix multiplication and add bias term
        #activate using tanh
        output1 = X.dot(self.W1) + self.b1
        output1 = np.tanh(output1)
        output2 = output1.dot(self.W2) + self.b2
        return output1, output2



    def backpropagate(self, X, Y, learningRate, regLambda, output1, softmax):
        
        #change of cost function (tanh) applied to weights individually
        delta3 = softmax
        delta3[range(len(X)),Y] -= 1
        dw2 = (output1.T).dot(delta3)
        db2 = np.sum(delta3, axis = 0, keepdims = True)
        delta2 = delta3.dot(self.W2.T) * (1 - np.power(output1, 2))
        dw1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis = 0)
            
        #change each weight impletmenting regLambda
        dw2 += regLambda * self.W2
        dw1 += regLambda * self.W1
        
        #Gradient descent and update weights
        self.W1 -= learningRate * dw1
        self.b1 -= learningRate * db1
        self.W2 -= learningRate * dw2
        self.b2 -= learningRate * db2
        
        
        
    def predict(self,X):

        #forward propogate to get current prediction
        output1, output2 = self.forward(X)
        eoutput2 = np.exp(output2)
        softmax = eoutput2 / np.sum(eoutput2, axis=1, keepdims=True)
        return np.argmax(softmax, axis=1)



def getData(inputData, dataLabels):
    '''
    Returns
    -------
    X : numpy matrix
        Input data samples.
    Y : numpy array
        Input data labels.
    '''
    # TO-DO for this part:
    # Use your preferred method to read the csv files.
    # Write your codes here:
    X = []
    Y = []
    with open(inputData) as csvDataFile:
        readLinData = csv.reader(csvDataFile)
        for row in readLinData:
            X.append([float(row[0]), float(row[1])])
    with open(dataLabels) as csvDataFile:
        readLinData = csv.reader(csvDataFile)
        for row in readLinData:
            Y.append(int(float(row[0])))
    X = np.asarray(X)
    Y = np.asarray(Y)
    # Hint: use print(X.shape) to check if your results are valid.
    return X, Y

def splitData(X, Y, K = 5):
    
    #split data for training and testing in terms of indices
    indices = []
    for i in range(len(X)):
        indices.append(i)
    shuffle(indices)
    dataSet = []
    #alternate list of train indices and test indices
    for i in range(K):
        stopInd = K - 1 - i
        holdSet = []
        holdSet += indices[0: int(stopInd * len(indices) / K)]
        holdSet += indices[int(stopInd * len(indices) / K + len(indices) / K): len(indices)]
        dataSet.append(holdSet)
        dataSet.append(indices[int(stopInd * len(indices) / K): int(stopInd * len(indices) / K + len(indices) / K)])
    '''
    Returns
    -------
    result : List[[train, test]]
        "train" is a list of indices corresponding to the training samples in the data.
        "test" is a list of indices corresponding to the testing samples in the data.
        For example, if the first list in the result is [[0, 1, 2, 3], [4]], then the 4th
        sample in the data is used for testing while the 0th, 1st, 2nd, and 3rd samples
        are for training.
    '''
    
    # Make sure you shuffle each train list.
    return dataSet

def plotDecisionBoundary(model, X, Y, Ws):
    """
    Plot the decision boundary given by model.
    Parameters
    ----------
    model : model, whose parameters are used to plot the decision boundary.
    X : input data
    Y : input labels
    """
    x1_array, x2_array = np.meshgrid(np.arange(-4, 4, 0.01), np.arange(-4, 4, 0.01))
    grid_coordinates = np.c_[x1_array.ravel(), x2_array.ravel()]
    Z = model.predict(grid_coordinates)
    Z = Z.reshape(x1_array.shape)
    plt.contourf(x1_array, x2_array, Z, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.bwr)
    plt.show()

def train(XTrain, YTrain, args):
    """
    This function is used for the training phase.
    Parameters
    ----------
    XTrain : numpy matrix
        The matrix containing samples features (not indices) for training.
    YTrain : numpy array
        The array containing labels for training.
    args : List
        The list of parameters to set up the NN model.
    Returns
    -------
    NN : NeuralNetwork object
        This should be the trained NN object.
    """
    

    learningRate = 0.01
    epochs = 5000
    regLambda = 0.01

    # 1. Initializes a network object with given args.
    NN = NeuralNetwork(args[0], args[1], args[2])

    
    
    # 2. Train the model with the function "fit".
    # (hint: use the plotDecisionBoundary function to visualize after training)
    NN.fit(X,Y, learningRate, epochs, regLambda)
    # 3. Return the model.
    
    return NN

def test(XTest, model):
    """
    This function is used for the testing phase.
    Parameters
    ----------
    XTest : numpy matrix
        The matrix containing samples features (not indices) for testing.
    model : NeuralNetwork object
        This should be a trained NN model.
    Returns
    -------
    YPredict : numpy array
        The predictions of X.
    """
    YPredict = model.predict(XTest)
    
    return YPredict

def getConfusionMatrix(YTrue, YPredict):
    """
    Computes the confusion matrix.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    CM : numpy matrix
        The confusion matrix.
    """
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(YTrue.shape[0]):
        if YTrue[i] == 1 and YPredict[i] == 1:
            TP += 1
        if YTrue[i] == 1 and YPredict[i] == 0:
            FN += 1
        if YTrue[i] == 0 and YPredict[i] == 1:
            FP += 1
        if YTrue[i] == 0 and YPredict[i] == 0:
            TN += 1

    CM = np.matrix([[TP, FN], [FP, TN]])       
    
    return CM
    
def getPerformanceScores(YTrue, YPredict):
    """
    Computes the accuracy, precision, recall, f1 score.
    Parameters
    ----------
    YTrue : numpy array
        This array contains the ground truth.
    YPredict : numpy array
        This array contains the predictions.
    Returns
    {"CM" : numpy matrix,
    "accuracy" : float,
    "precision" : float,
    "recall" : float,
    "f1" : float}
        This should be a dictionary.
    """
    CM = getConfusionMatrix(YTrue, YPredict)
    accuracy = (CM[0, 0] + CM[1, 1]) / (CM[0, 0] + CM[0, 1] + CM[1, 0] + CM[1, 1])
    precision = CM[0, 0] / (CM[0, 0] + CM[1, 0])
    recall = CM[0, 0] / (CM[0, 0] + CM[0, 1])
    f1 = (2 * precision * recall) / (precision + recall)
    
    return {"CM" : CM, "accuracy" : accuracy, "precision" : precision, "recall" : recall, "f1" : f1}


X, Y = getData("NonlinearX.csv", "NonlinearY.csv")
K = 5

dataSet = splitData(X, Y, K)

NNodes = 10

input_dim = X.shape[1] 
output_dim = 2
    
NNargs = [input_dim, output_dim, NNodes]

#use 4 groups and train 1 group change train and test groups until used all groups as test
for i in range(0, 2 * K, 2):    
    NN = train(X[dataSet[i]], Y[dataSet[i]], NNargs)
    test(X[dataSet[i + 1]], NN)
    Ws = [NN.W1, NN.W2]
    plotDecisionBoundary(NN, X[dataSet[i]], NN.predict(X[dataSet[i]]), Ws)
    plt.title("Neural Net Decision Boundary")
    print(getPerformanceScores(Y[dataSet[i]], NN.predict(X[dataSet[i]])))  
    