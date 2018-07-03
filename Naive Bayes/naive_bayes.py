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

def data_analysis(data, labels):
	'''
	This function is to analyze data
	Obtain value sets of attributes
	[class, attribute, values]
	'''
	#data, labels = train_set['data'], train_set['labels']
	value_sets = []

	num_class = len(set(labels))
	data_dim = len(data[0])

	for i in range(num_class):
		tmp_class = []
		for i in range(data_dim):
			tmp_set = np.asarray(set(data[:, i]))
			tmp_class.append(tmp_set)
		value_sets.append(tmp_class)

	return value_sets, num_class, data_dim


def train_NB_classifier(train_set, value_sets):
	# count numbers of samples in dataset
	data, labels = train_set['data'], train_set['labels']
	value_sets, num_class, data_dim = data_analysis(data, labels)

	# create zero arrays for prior and conditional probability
	prior_prob = np.zeros(num_class)
	# con_prob = np.zero((num_class, data_dim))

	# traverse all samples in dataset
	for i, feature in enumerate(data):

		category = labels[i]
		prior_prob[category] += 1

		class_values = value_sets[category]

		for j in range(data_dim):
			# 这里需要统计各个属性的可能取值, 并分别计算条件概率用于累乘
			index = np.where(class_values[j] == feature[j])
			

if __name__ == '__main__':
	train_set, test_set = load_data()
	value_sets = data_analysis(train_set['data'], train_set['labels'])
	print(value_sets)
	