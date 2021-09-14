import numpy as np
import math

class Network:
    def __init__(self,name,inSize,outSize,neurons,genetic=False,gw=None,gb=None):
        self.name = name
        self.y = [None]*(len(neurons)+2)
        self.a = [None]*(len(neurons)+2)
        self.w = []
        self.b = []
        if(not genetic):
            sd = 0.5
        else:
            sd = 0.001
        self.lr = 0.01
        #0.035
        self.w.append(np.random.normal(0,sd,(neurons[0],inSize)))
        self.b.append(np.random.normal(0,sd,(neurons[0])))
        for x in range(1,len(neurons)):
            self.w.append(np.random.normal(0,sd,(neurons[x],neurons[x-1])))
            self.b.append(np.random.normal(0,sd,(neurons[x])))
        self.w.append(np.random.normal(0,sd,(outSize,neurons[-1])))
        self.b.append(np.random.normal(0,sd,(outSize)))

        if(genetic):
            for wx in range(len(self.w)):
                self.w[wx] = self.w[wx] + gw[wx]

            for bx in range(len(self.b)):
                self.b[bx] = self.b[bx] + gb[bx]
    def calculate(self,inputVector):
        self.y[0] = inputVector
        for x in range(0,len(self.b)):
            self.a[x+1] = np.add(self.b[x],np.matmul(self.w[x],self.y[x]))
            self.y[x+1] = self.sigmoid(self.a[x+1])
        return(self.y[-1])

    def backpropagate(self,yP):
        errors = []
        errUp = self.errorOutput(yP)
        errors.append(errUp)
        k = len(self.w)-1
        for x in range(k):
            errUp = self.errorNeuron(k-x,errUp)
            errors.insert(0,errUp)
        for y in range(1,k+2):
            n = -1*y
            self.b[n] = self.b[n] - self.lr*errors[n]
            self.w[n] = self.w[n] - self.lr*np.outer(errors[n],self.y[n-1])

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def sigprime(self,z):
        t = self.sigmoid(z)
        return t*(1-t)

    def QuadraticCPrime(self,yP):
        return self.y[-1] - yP

    def errorOutput(self,yP):
        return self.sigprime(self.a[-1])*self.QuadraticCPrime(yP)

    def errorNeuron(self,x,errUp):
        #print(np.matmul(self.w[x].transpose,errUp))
        return self.sigprime(self.a[x])*np.matmul(self.w[x].transpose(),errUp)

    def toProbability(self):
        sum = np.sum(np.exp(self.y[-1]))
        self.y[-1] = np.exp(self.y[-1])/sum

    def teach(self,inData,outData):
        for x in range(len(inData)):
            self.calculate(inData[x])
            #print("actual: ", self.y[-1], " ; expected: ", outData[x], " ; cost: ",np.sum(np.square(self.y[-1] - outData[x])))
            self.backpropagate(outData[x])

    def onTheFlyTeach(self,inData,outData):
        r = self.calculate(inData)
        self.backpropagate(outData)
        return r

    def test(self,inData,outData):
        s = 0
        for x in range(len(inData)):
            y = self.calculate(inData[x])
            if(np.argmax(y)==np.argmax(outData[x])):
                s += 1
        return 100*(s/len(inData))

    def save(self,filename):
        outfile = open(filename,'w')
        outfile.write(self.name + "\n")
        outfile.write(str(len(self.w)) + "\n")
        for x in self.w:
            for y in x:
                for z in y:
                    outfile.write(str(float(z)) + ",")
                outfile.write("|")
            outfile.write("\n")
        outfile.write("\n")

        for x in self.b:
            for y in x:
                outfile.write(str(float(y)) + ",")
            outfile.write("\n")

    def load(self,filename):
        infile = open(filename,'r')
        self.name = infile.readline().rstrip("\n")
        wlen = int(infile.readline().rstrip("\n"))
        self.w = []
        self.b = []
        for x in range(wlen):
            wi = infile.readline().rstrip("\n")[:-2].split("|")
            wCol = []
            for y in wi:
                temp = y[:-1].split(",")
                wRow = []
                for z in temp:
                    wRow.append(float(z))
                wCol.append(wRow)
            self.w.append(np.array(wCol))

        infile.readline()

        for x in range(wlen):
            bi = infile.readline().rstrip("\n")[:-1].split(",")
            bCol = []
            for y in bi:
                bCol.append(float(y))
            self.b.append(np.array(bCol))
