from sklearn.linear_model import LinearRegression
import numpy as np

import DataSet

class Model:
	
	# This should build inverse models and it will assume that all data is perfekt, and from
	# this it can make interpolations and extrapoations.
	# 
	#	n_neighbors - how many neighbors to take into account
	#	alpha 		- noise level
	#	remember 	- how much of a memory is lost when called. 1 = perfect memory
	
	def __init__(self, n_neighbors = 10, alpha = 10**(-3), remember = 1):
		
		self.DS    		 = DataSet.DataSet(n_neighbors, remember = remember)
		self.alpha 		 = alpha
		self.Reg   		 = LinearRegression()
		self.dist  		 = []	# Distance from prediction over time
		
		
	# STATISTICS #
	# Return the mean and the covariance matrix for one sample.
	# x is the position we want to approximate f for.
	def get_distribution(self, x, keep_mem = False):
		
		# Get neighbors
		X, F, dist = self.DS.get_neighbours(x, keep_mem)
		
		# Fit to neighbors
		self.Reg.fit(X, F) #, 1./dist)

		# Compute mean and covariance
		mu  = self.Reg.predict(x.reshape(1,-1))[0] # np.mean(F, 0) #
		
		if len(mu) > 1:
			if len(F) > 1:
				cov	= np.cov(F.T) + self.alpha*np.eye(len(mu))
			else:
				cov	= self.alpha*np.eye(len(mu))
		else:
			var = np.cov(F.T) + self.alpha
			cov = np.array([[var]])
	
		cov *= np.exp(min(dist)/.05)
	
		return mu, cov

	
	# Give prediction of f given x without variance
	def predict(self, x):
		
		mu, _ = self.get_distribution(x)
		mu.clip(0,1)

		return mu
		
	
	# Gives samples of f at x within destribution of seen data, for X = [x1,x2,...]
	# This is basically a prediction with sensible noise.
	def sample_y(self, x, noise = 1., n_samples = 1, keep_mem = False):
		
		mu, cov = self.get_distribution(x, keep_mem)
		L = np.linalg.cholesky(cov)
		
		DoF = len(mu)
		noise *= np.random.rand()
		Y_approx = mu + noise*np.matmul(np.random.randn(n_samples, DoF), L.T)

		return Y_approx.clip(0,1)
		
		
	# TRAINING #
	# Train the regressor using samples X = [x1, x2, ....], F = [f1, f2, ...]
	def fit(self, X, F, pres = 1):
		
		n_samples = len(X)
		precision = pres*np.ones(n_samples)	# New and fresh in mind
		self.DS.add_data(X, F, precision)
		
		
	def get_n_samples(self):
		return self.DS.n_data

