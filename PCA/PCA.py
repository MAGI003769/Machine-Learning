import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

def load_data():
	# load iris dataset from sklearn api
	# dictionary-like object which has atrributes
	# 'data', 'target', 'feature_names', 
	# 'DESCR' to see descrption of dataset
	iris = datasets.load_iris()

	return iris.data, iris.target


def my_PCA(raw_data, output_dim):
	# x = np.transpose(x)
	T = raw_data - np.mean(raw_data, axis=0)
	cov_mat = np.cov(T.transpose())
	w, v = np.linalg.eig(cov_mat)
	reduced_x = np.matmul(raw_data, v[:, :output_dim])
	return reduced_x


def scatter_plot(data, label, title):
	cat0_x, cat0_y = [], []
	cat1_x, cat1_y = [], []
	cat2_x, cat2_y = [], []
	
	for i, feature in enumerate(data):
		if label[i] == 0:
			cat0_x.append(feature[0])
			cat0_y.append(feature[1])
		elif label[i] == 1:
			cat1_x.append(feature[0])
			cat1_y.append(feature[1])
		else:
			cat2_x.append(feature[0])
			cat2_y.append(feature[1])

	plt.scatter(cat0_x, cat0_y, c='r')
	plt.scatter(cat1_x, cat1_y, c='g', marker='D')
	plt.scatter(cat2_x, cat2_y, c='b', marker='^')
	plt.grid()
	plt.title(title)



if __name__ == '__main__':
	iris_X, iris_Y = load_data()
	reduced_X = my_PCA(iris_X, 2)
	sklearn_pca = PCA(2)
	sklearn_X = sklearn_pca.fit_transform(iris_X)

	plt.subplot(1,2,1)
	scatter_plot(reduced_X, iris_Y, 'my_PCA')

	plt.subplot(1,2,2)
	scatter_plot(sklearn_X, iris_Y, 'sklearn_PCA')

	plt.show()
	
