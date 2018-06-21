'''
Author: Ruihao Wang
Description: Implementaiton of naive bayes classifier

f(x) = argmax (prior*likelyhood)
'''

import numpy as np
from sklearn import datasets


def load_data():
	# load iris dataset from sklearn api
	# dictionary-linke object which has atrributes
	# 'data', 'target', 'feature_names', 
	# 'DESCR' to see descrption of dataset
	iris = datasets.load_iris()

	# create empty set
	train_set = {}
	test_set = {}
	offset = 50
	num_train = 45

	# In fact, we can use np.vsplit / np.hsplit

	# split features
	train_set['data'] = np.vstack([iris.data[0:num_train], 
									iris.data[0+offset:num_train+offset], 
									iris.data[0+2*offset:num_train+2*offset]])

	test_set['data'] = np.vstack([iris.data[num_train:offset], 
									iris.data[num_train+offset:2*offset], 
									iris.data[num_train+2*offset:]])

	# split labels
	train_set['labels'] = np.hstack([iris.target[0:num_train],
										iris.target[0+offset:num_train+offset],
										iris.target[0+2*offset:num_train+2*offset]])

	test_set['labels'] = np.hstack([iris.target[num_train:offset], 
										iris.target[num_train+offset:2*offset], 
										iris.target[num_train+2*offset:]])

	return train_set, test_set

def train_NB_classifier(data, labels):
	# count numbers of samples in dataset
	num_samples = len(labels)
	num_class = len(set(lables))

	# dimensionality of data
	data_dim = len(data[0])

	# create zero arrays for prior and conditional probability
	prior_prob = np.zeros(num_class)
	con_prob = np.zero((num_class, data_dim))

	# traverse all samples in dataset
	for i, feature in enumerate(data):
		category = labels[i]

		prior_prob[category] += 1

		for j in range(data_dim):
			# 这里需要统计各个属性的可能取值
			# 并分别计算条件概率用于累乘

	# calculate prior and conditional probability

if __name__ == '__main__':
	train_set, test_set = load_data()
	print(test_set['data'])