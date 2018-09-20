from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import numpy as np

import DataSet

class Model:
	
	# This should build inverse models and it will assume that all data is perfekt, and from
	# this it can make interpolations and extrapoations.
	# 
	#	n_neighbors - how many neighbors to take into account
	#	alpha 		- noise level
	#	remember 	- how much of a memory is lost when called. 1 = perfect memory
	
	def __init__(self):
		
		self.DS    		 = DataSet.DataSet() #n_neighbors, remember = remember)
		self.Reg   		 = LinearRegression()
		#self.dist  		 = []	# Distance from prediction over time
		
	
	# Gives samples of y at x within destribution of seen data, for X = [x1,x2,...]
	def sample_y(self, x, noise = 1., use_mean = False):
		
		# Get neighbors
		X, F, dist = self.DS.get_data(x) #neighbours(x, keep_mem)
		
		
		noise *= np.random.rand() 
		#print('Final noise: ' + str(noise))
		
		if use_mean:
			#i_min 	= np.argmin(dist)
			mu 		= np.mean(F,0)	# F[i_min]	#np.mean(F,0)	# No regression
		else:
			dist /= np.max(dist)
			weight = 1./(dist + .01)	# Added small term to allow avaraging even close to sample
			
			self.Reg.fit(X, F, weight)

				
			mu = self.Reg.predict(x.reshape(1,-1))[0]
			
			x_mean = np.mean(X,0)
			f_mean = np.mean(F,0)
			
			# be careful if extrapolation if confidence < 1
			confidence = self.DS.max_dist/np.mean(dist)
			
			if confidence < 1. :
				mu = f_mean
			
		#mu  = (1-noise)*mu + noise*(2*np.random.rand(len(F[0]))-1)
		mu  += noise*np.random.randn(len(F[0]))
		
		if False: # plot
			for x0 in X:
				plt.plot(x0[0], x0[1], '.')
			
			plt.plot(x[0], x[1], '*')
			plt.axis([-1,1,-1,1])
			plt.show()
				
			for f in F:
				plt.plot(f)
				
			plt.plot(mu, 'k')
			plt.show()
		
		#print(mu)
		tmp = .5
		return [mu.clip(-tmp*np.pi,tmp*np.pi)]

		
	# TRAINING #
	# Train the regressor using samples (x,f)
	def fit(self, x, f):
		
		self.DS.add_data(x, f) #, precision)
		
		
	def get_n_samples(self):
		return self.DS.n_data
		

if False:
	inv = Model()
		
	x = 0.01*np.random.randn(10,2) + np.array([.2,.5])
	print(x)
	f = np.array([np.cos(x[:,0] + x[:,1]), np.sin(x[:,0] + x[:,1]) ]).T # [cos(x+y), sin(x+y)]
		
	inv.fit(x,f)
		
	print(inv.sample_y(np.array([0.2,0.5])))
	print(inv.sample_y(np.array([0.5,0.2])))
	print(inv.sample_y(np.array([0.1,0.5])))
	print(inv.sample_y(np.array([0.5,0.1])))


