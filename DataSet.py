# Dataset2
# This dataset stores N samples and forgets older ones after that.
# Returns the full datset together with the distances to a given point
# x.

import numpy as np

class DataSet:

	def __init__(self, data_size = 100):
	
		self.n_data    = 0			# Given datapoints including removed points
		self.data_size = data_size	# How many datapoints that may be saved
		self.x_data = []			# The dataset of x_values
		self.f_data = []			# The dataset of corresponding y_values
		
		self.max_dist = 0	# Max distance between two points in set
		
	
	# When adding new data, old data might be forgotten
	# This means that forgetting only happens when new data is added.
	def add_data(self, x, f):
		
		# Check maximum distance between points in set
		for x0 in self.x_data:
			distance = np.linalg.norm(x0-x)
			if distance > self.max_dist:
				self.max_dist = distance
			
		# Add new samples
		if False: #self.n_data > self.data_size:
			self.x_data.pop(0)
			self.f_data.pop(0)
			
		else:
			self.n_data += 1
			
		self.x_data.append(x)
		self.f_data.append(f)
		
			
		#i_sample = np.random.randint(self.data_size)
		#	
		#self.x_data[i_sample] = x
		#self.f_data[i_sample] = f


	# Return all X and F, together with distance to x
	def get_data(self, x):
		#print('Inv model has ' + str(self.n_data) + ' samples')
		
		X = np.array(self.x_data)
		F = np.array(self.f_data)
		D = X - x
		
		dist = np.linalg.norm(D, axis = 1)
		
		return X,F,dist
		
'''
D = DataSet()

for i in range(100):
	D.add_data(np.random.rand(20,3),np.random.rand(20,5))
	print([len(D.x_data), D.n_data])
	
	x,f,d = D.get_data(np.array([.5,.5,.5]))

'''
