import numpy as np
import sys,csv,random

class RandomForestLearner:
    def __init__(self,k):
        self.k = k
        self.index = 0
        self.forest = []
        
    def addEvidence(self,data):
        for i in range(self.k):
            self.index = 0
            self.forest.append(self.buildTree(data))
            
    def buildTree(self,data):
        index = self.index
        self.index = self.index + 1
        cols = len(data[0,:])

        #if only one node add as leaf node
        if(len(data)==1):
            return np.array([index,-1,data[0][cols-1],-1,-1])
        
        left = np.zeros((0,cols))
        right = np.zeros((0,cols))
        #random selection of feature
        rand_feature = random.choice(range(cols-1))
        feature = data[:,rand_feature]
        
        #checking if data can be split
        flag = 0
        for i in range(1,len(feature)):
            if feature[i] != feature[0]:
                flag = 1
        if flag == 0:
            #data cant be split so add as leaf with mean of the feature values as value
            return np.array([index,-1,np.mean(data[:,cols-1]),-1,-1])

        #choosing split value
        split = np.mean(random.sample(feature,2))
        
        for i in range(len(data)):
            #if value is <= split value add node to left
            if (data[i,rand_feature] <= split):
                left = np.vstack((left,data[i]))
            #if value is > split value add node to right
            else:
                right = np.vstack((right,data[i]))

        #building tree to left and right of node
        ltree = self.buildTree(left)
        rtree = self.buildTree(right)

        #finding left and right indices of node
        l_index = -1
        if ltree.ndim == 1:
            l_index = ltree[0]
        else:
            l_index = ltree[0][0]
        
        r_index = -1
        if rtree.ndim == 1:
            r_index = rtree[0]
        else:
            r_index = rtree[0][0]

        #adding node,left and right to tree 
        node = np.array([index,rand_feature,split,l_index,r_index])
        return np.vstack((node,ltree,rtree))
            
    def query(self,data):  
        n = len(data)
        Y = np.zeros((n))

        for i in range(n):
            x = np.zeros((self.k))
            # for set of trees
            for k in range(self.k):
                tree = self.forest[k]
                curr_node = 0
                while(tree[curr_node][1] != -1):
                    #if value is <= split value move left
                    if (data[i][tree[curr_node][1]] <= tree[curr_node][2]):
                        curr_node = tree[curr_node][3]
                    #if value is > split value move right
                    else:
                        curr_node = tree[curr_node][4]
                x[k] = tree[curr_node,2]
            #mean of resultant values from trees
            Y[i] = np.mean(x)
        return Y                 
