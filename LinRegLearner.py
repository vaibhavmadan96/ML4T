import numpy as np
import sys,csv,math

class LinRegLearner:
    def addEvidence(self, Xtrain, Ytrain):
        Xtrain = np.vstack([Xtrain[:,0], Xtrain[:,1], np.ones(len(Xtrain))]).T
        #[[x 1]]
        #vertically stack arrays        
        res = np.linalg.lstsq(Xtrain, Ytrain)
        #Return the least-squares solution to a linear matrix equation
        #returns m,c
        self.train = res[0]
        #self.train2=res[1]
        #shared by all members

    def query(self, Xtest):
        Xtest = np.vstack([Xtest[:,0], Xtest[:,1], np.ones(len(Xtest))]).T
        res = np.dot(Xtest,self.train)
        #Dot product of two arrays
        #res+=self.train2
        return res

