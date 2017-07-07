'''
This is a dataset. 
- It stores x and f data. 
- It adds one dimension to x that reflects how reliable the point is.
- It returns neighborhoods to a point where reliability in included
- Every reliable point starts at 1 (the rest at 0), but then decays proportionally to how many
	new points it been called to, assuming that every requested batch leads to one new data-point.
	This mechanism is dangerous since it punishes points for being used, even though they might
	not lead to a new good point. However, if a point is unhelpful on avarage, maybe its not
	that bad if it's forgotten?
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors

class DataSet:
	
	def __init__(self, n_neighbors = 10, remember = 1, batchsize = 100):
	
		self.n_data = 0
		self.x_data = []	# 3D, in this a distance to the solution manifold should be included
		self.f_data = []
		self.n_neighbors = n_neighbors
		self.remember    = remember**(1./n_neighbors)	# Decay of memories for a call
		self.nhood 		 = NearestNeighbors(algorithm='brute')	# The neighborhood
		self.batchsize	 = min(batchsize, n_neighbors) 		#How many new samples to take in beforeupdating
		self.memory_max  = 10000			# Reduce to half if this happens
		
		
	# Precision tells how much we trust the point.
	def add_data(self, X, F, precision):
		
		n_samples, n_dim = X.shape
		self.n_data     += n_samples
		
		for i in range(n_samples):
			x_extended = np.append(X[i], precision[i])
			
			self.x_data.append(x_extended)
			self.f_data.append(F[i])

		# Update in batches. This assumes that the data base gets one sample at a time
		if self.n_data < self.n_neighbors or np.mod(self.n_data, self.batchsize) == 0:
			self.nhood.fit(np.array(self.x_data))
			
		
	# Return neighbours to one point x, together with their function values.
	def get_neighbours(self, x, keep_mem = False):
		n_dim = len(x)
		
		x_extended = np.append(x, 1.).reshape(1, -1)
		
		dist, indices = self.nhood.kneighbors(x_extended, \
										n_neighbors = min(self.n_neighbors, self.n_data))
		
		X = []
		F = []
		
		for i in indices.flatten():
			
			X.append(self.x_data[i][0:n_dim])
			F.append(self.f_data[i])
			
			if not keep_mem:
				self.x_data[i][n_dim] *= self.remember	# The more a memory is used, the more it
														# is forgotten. If it's a good memory it
														# will help create multiple similar memories.
			
		return np.array(X), np.array(F), dist[0]
		

		

####################################################################################

if False:
	n_data = 10	# How many neighbours
	batch  = 20

	D = DataSet(n_data)

	f = lambda x: (x**2)

	for i in range(100):
		x_data      = np.random.rand(batch,1)
		f_data		= f(x_data)
		prec 		= np.ones(batch)
		
		D.add_data(x_data, f_data, prec)

		x_tmp, y_tmp, dist = D.get_neighbours(np.array([.5]))
		
	for d in D.x_data:
		print(d)
