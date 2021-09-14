import numpy as np
import math
import random
import pandas as pd
import nnMatrix as nnM

def loadCSV(path,inputCount,outTypes):
    df = pd.read_csv(path)#.sample(frac=1)
    inData = df.values.tolist()
    outData = []
    for z in range(len(inData)):
        outData.append(inData[z].pop(-1))
    outData = outDataBinaryConverter(outData,outTypes)
    inDataFinal = []
    outDataFinal = []
    for x in range(len(inData)):
        inDataFinal.append(np.array(inData[x]))
        outDataFinal.append(np.array(outData[x]))
    return inDataFinal,outDataFinal

def outDataBinaryConverter(outData,outTypes):
    newOutData = []
    for x in range(len(outData)):
        ind = 0
        for i in range(len(outTypes)):
            if(outTypes[i] in str(outData[x])):
                ind = i
        newOutData.append([0]*len(outTypes))
        newOutData[x][ind] = 1
    return newOutData


#IRIS CLASSIFICATION
# epochs = 200
# outLabels = ['versicolor','setosa','virginica']
# inTrain,outTrain = loadCSV('iris_flowers.csv',4,outLabels)
# inTest,outTest = loadCSV('iris_testing.csv',4,outLabels)
# nn = nnM.Network('iris classification',4,3,[2])
# for x in range(epochs):
#     nn.teach(inTrain,outTrain)
#     print(nn.test(inTest,outTest),"%")

#MNIST HANDWRITTEN DIGITS
# epochs = 100
# outLabels = ['0','1','2','3','4','5','6','7','8','9']
# inTrain,outTrain = loadCSV('mnist_train.csv',4,outLabels)
# inTest,outTest = loadCSV('mnist_test.csv',4,outLabels)
# print('data in')
# nn = nnM.Network('iris classification',784,10,[200,80])
# for x in range(epochs):
#     nn.teach(inTrain,outTrain)
#     print(nn.test(inTest,outTest),"%")

#MNIST clothing images
epochs = 100
outLabels = ['0','1','2','3','4','5','6','7','8','9']
inTrain,outTrain = loadCSV('fashion-mnist_train.csv',4,outLabels)
inTest,outTest = loadCSV('fashion-mnist_test.csv',4,outLabels)
nn = nnM.Network('iris classification',784,10,[100,40])
for x in range(epochs):
    nn.teach(inTrain,outTrain)
    print(nn.test(inTest,outTest),"%")
