import numpy as np
import sys, csv, math
class KNNLearner:
    def __init__(self,k):
        self.k = k

    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain

    def query(self,Xtest):
        n_test = len(Xtest)
        n_train = len(self.Xtrain)
        Y = np.zeros(n_test)
        for i in range(n_test):#pick one test pt
            dist = []
            for j in range(n_train):#check for all train pts
                d = 0
                for k in range(len(Xtest[i])):#takin into account all cols calc dist
                    diff = Xtest[i][k] - self.Xtrain[j][k]
                    d += math.pow(diff,2)
                dist.append((math.sqrt(d),j))#fill dist list wtih subs dist
            dist.sort()
            neighbours = []#consider first k neightbours
            for k in range(self.k):
                try:
                    neighbours.append(self.Ytrain[dist[k][1]])
                except IndexError:
                    pass
            Y[i] = np.mean(neighbours)
        return Y
