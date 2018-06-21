import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cov_mat = [[0.09, 0.08], [0.08, 0.09]]

def data_generation():
    '''
    Generate two groups of data based on 2D normal distribution
    
    Returns
    -------
    X: data as features for training
    y: labels has value 0 or 1
    '''
    n, dim = 100, 2
    pos = np.random.multivariate_normal(mean=[1, 1], cov=cov_mat, size=n)
    neg = np.random.multivariate_normal(mean=[1.5, 0.5], cov=cov_mat, size=n)
    X = np.r_[pos, neg]
    y = np.hstack((np.zeros(n), np.ones(n))).astype(np.int)
    print(y.shape)
    return X, y

class Perceptron(object):
    
    def __init__(self, learning_rate, epoch):
        self.learning_rate = learning_rate
        self.max_iteration = epoch
        
    
    def train_all(self, features, labels):
    	'''
        Arguments
        ---------
        features: (N, n) numpy array, N samples of n-dimension feature
        labels: (N, ) numpy array, N labels

        Returns
        -------
        None, only modify attributes of perceptron
        '''

        self.w = np.random.randn(features[0].shape[0] + 1) # initialize weight matrix w with bias b
        
        for epoch in range(self.max_iteration):
            bingo = 0
            for i in range(len(features)):
                x = np.hstack((features[i], np.ones(1)))
                y = 2 * labels[i] - 1
                wx = sum(self.w * x)       # calculate wX_i + b
                
                if wx * y > 0:
                    bingo += 1
                else:
                    self.w += self.learning_rate * (y * x)
            print("Epoch {}, Bingo {}".format(epoch+1, bingo))
                
    
    def predict_one(self, x):
        return int( sum(self.w * x) >0 )
        
    
    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_one(x))
        return labels

class dual_Perceptron(object):
    
    def __init__(self, learning_rate, epoch):
        self.learning_rate = learning_rate
        self.max_iteration = epoch
    
    def dual_train(self, features, labels):
        
        self.alpha, self.b = np.zeros(features.shape[0]), 0
        
        gram = []
        for i in range(len(features)):
            temp = [np.dot(features[i], features[j]) for j in range(len(features))]
            gram.append(temp)
        
        print(len(gram[0]))
        
        for epoch in range(self.max_iteration):
            bingo = 0
            for i in range(len(features)):
                y_i = 2 * labels[i] - 1
                cal = (np.sum(self.alpha * labels * gram[i]) + self.b) * y_i
                if cal > 0:
                    bingo += 1
                else:
                    self.alpha[i] += self.learning_rate
                    self.b += + self.learning_rate * y_i
            print("Epoch {}, Bingo {}".format(epoch+1, bingo))
    
    def predict_one(self, x, features, labels):
        temp_dot = np.array([np.dot(features[i], x) for i in range(len(features))])
        return int(np.sum(self.alpha * labels * temp_dot) + self.b > 0)
    
    def predict(self, x):
        labels = []
        for feature in features:
            labels.append(self.predict_one(feature))
        return labels

X, y = data_generation()      # generate 2D normal distributed data 
p = Perceptron(0.0005, 25)    # initialize perceptron
p.train_all(X, y)             # train model

# visualize the result
plt.scatter(X[y==0,0], X[y==0, 1], c = 'r', edgecolor='k', s=15)
plt.scatter(X[y==1,0], X[y==1, 1], c = 'b', edgecolor='k', s=15)
plain_x1 = np.linspace(-0.1, 2.5, 100)
plain_x2 = - (p.w[2] + p.w[0] * plain_x1) / p.w[1]
plt.plot(plain_x1, plain_x2, 'k-')
print(p.w)