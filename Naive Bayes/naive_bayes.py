'''
@author: Ruihao Wang
Description: Implementaiton of naive bayes classifier

f(x) = argmax (prior*likelyhood)

API reference: http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes
'''

import numpy as np
from sklearn import datasets

def load_data_sep():
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

def load_data():
	iris = datasets.load_iris()
	X = iris.data
	Y = iris.target
	return X, Y



class MultinomialNB(object):

	def __init__(self, lambda_=0.5, fit_prior=True, class_prior=None):
		self.lambda_ = lambda_
		self.fit_prior = fit_prior
		self.class_prior = class_prior
		self.classes = None
		self.conditional_prob = None

	def _feature_prob(self, features):
		total_num = float(len(features))
		values = np.unique(features)
		value_num = float(len(values))
		value_prob = {}
		for v in values:
			value_prob[v] = (np.sum(np.equal(features, v)) + self.lambda_) / (total_num + value_num * self.lambda_)
		return value_prob

	def fit(self, X, Y):

		self.classes = np.unique(Y)

		# 计算类别先验 P(Y=c_k)
		if self.class_prior == None:
			class_num = float(len(self.classes))
			if not self.fit_prior:
				self.class_prior = [1.0/class_num for _ in range(class_num)] # uniform distribution
			else:
				self.class_prior = []
				sample_num = float(len(Y))
				for c in self.classes:
					c_num = np.sum(np.equal(Y, c))
					self.class_prior.append((c_num + self.lambda_)/(sample_num + class_num*self.lambda_))

		# 计算条件概率(likelyhood)
		self.conditional_prob = {}
		for c in self.classes:
			self.conditional_prob[c] = {}
			for i in range(X.shape[1]):
				features = X[np.equal(Y, c)][:, i]
				self.conditional_prob[c][i] = self._feature_prob(features)
		return self

	# ????
	def _get_xj_prob(self, values_prob, target_value):
		return values_prob[target_value]

	def _predict_one(self, x):

		# 需要对最终的后验概率及返回结果做初始化
		label = 0
		max_posterior_prob = 0

		# 对每个类别分别计算后验概率
		for c_index in range(len(self.classes)):
			current_class_prior = self.class_prior[c_index]
			current_conditional_prob = 1.0
			feature_prob = self.conditional_prob[self.classes[c_index]]
			j = 0
			for feature_i in feature_prob.keys():
				print('Current feature_prob: \n', feature_prob[feature_i])
				current_conditional_prob *= self._get_xj_prob(feature_prob[feature_i], x[j])
				j += 1
			print('Class i over !!!')
			if current_class_prior * current_conditional_prob > max_posterior_prob:
				max_posterior_prob = current_class_prior * current_conditional_prob
				label = self.classes[c_index]
		return label

	def predict(self, X):
		if X.ndim == 1:
			return self._predict_one(X)
		else:
			labels = []
			for i, sample in enumerate(X):
				print('This {}th sample.'.format(i))
				lable = self._predict_one(sample)
				labels.append(label)
			return labels

if __name__ == '__main__':
	X, Y = load_data()
	print(X)
	print(Y)
	nb = MultinomialNB()
	nb.fit(X, Y)
	prediction = nb.predict(X[1])
	