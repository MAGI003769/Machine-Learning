import numpy as np

class GaussianFeatures(object):
	'''
	Gaussian features

	gaussian function = exp(-0.5 * (x - m) / v)

	where m is mean and v is variance
	'''

	def __init__(self, mean, var):
		'''
		construct gaussian features

		Parameters
		----------
		mean : (n_features, ndim) or (n_features,) ndarray
		    the places where gaussian function locate
		var : float
		    variance of gaussian funciton
		'''
		if mean.ndim == 1:
			mean = mean[:, None]
		else:
			assert mean.ndim == 2
		assert isinstance(var, float) or isinstance(var, int)
		self.mean = mean
		self.var = var

	def _gauss(self, x, mean):
		return np.exp(-0.5 * np.sum(np.square(x - mean), axis=-1) / self.var)

	def transform(self, x):
		'''
		transform input array with gaussian features

		Parameters
		----------
		x : (sample_size, ndim) or (sample_size,)
		    input array

		Returns
		-------
		output : (sample_size, n_features) gaussian features
		'''
		if x.ndim == 1:
			x = x[:, None]
		else:
			assert x.ndim == 2
		assert np.size(x, 1) == np.size(self.mean, 1)
		basis = []
		for m in self.mean:
			basis.append(self._gauss(x, m))
		return np.asarray(basis).transpose()
