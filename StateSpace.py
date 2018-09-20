import numpy as np
import Inverse
from tqdm import *

class Parent(object):
	# (mem_decay = mem_decay, max_noise = max_noise)

	def __init__(self, n_states, state_coord, max_noise = .1, diff_max = 1.5):
		
		self.n_states = n_states	# Number of states
		
		self.s		 = -1		# Current state
		self.s_last	 = -1		# last state != -1
		self.region  = -1
		self.actions = []
		self.states  = [None]*n_states # An inv-model for every state action
		
		self.learn	  = True
		self.diff_max = diff_max
		
		# Initialization
		# - Create states
		self.X = state_coord	# Coordinates of states
					
		# - Acquire actions list
		for s in range(self.n_states):
			As = []		# Possible actions for state s
			
			for a in range(self.n_states):
				
				if self.are_neighbours(s, a):
					As.append(a)
			
			self.actions.append(As)
			
	# END INIT
		

	# PUBLIC
	# Incorprate seen data to world model
	def observe(self, x, q):
		s_i = self.s
		
		# FIGURE OUT s_f
		region 		= self.get_s(x)
		self.region = region
		
		s_f 	= region
		failed 	= False
		
		# Sucessful transition?
		if self.states[region] is not None:	# If region has been successfully visited

			q_aprx1 = self.states[region].sample_y(x, noise = 0, use_mean = False)[0]
			q_aprx2 = self.states[region].sample_y(x, noise = 0, use_mean = True)[0]
			
			diff1 = np.max(np.abs(q_aprx1 - q))
			diff2 = np.max(np.abs(q_aprx2 - q))
			
			diff  = min(diff1, diff2)
			
			print('Diff: ' + str(np.round(np.abs(q_aprx1 - q),3)))
			
			str_diff = str(np.round([diff1,diff2],3))
			if diff < self.diff_max:
				print('Success with diff ' + str_diff)
				
			else:
				print('Failed with diff ' + str_diff)
				s_f	   = -1
				failed = True
				
		# If a new region was visited from off-manifold posture and not first iteration
		elif False: #s_i == -1 and self.s_last != -1: 
			s_f	   = -1
			failed = True
			
		else:
			print('State (' + str(s_f) + ') was found!')


		# Update memories
		self.s = s_f

			
		# TRAIN INVERSE MODELS WITH x,q
		if s_f != -1:
				
			self.s_last	= s_f
							
			if self.learn:
				
				if self.states[s_f] is None:
					self.states[s_f] = Inverse.Model()
					
				self.states[s_f].fit(x, q)
		
		#self.x = x
		#self.q = q


	# Return x,q to reach s_g,x_G
	def get_goal(self, s_g, x_G = None, noise = 1.0, use_mean = False):
		
		if x_G is None or not self.state_contains(s_g, x_G):
			x = self.sample_x(s_g)
		else:
			x = x_G

		# What inverse model to use?
		if self.states[s_g] is not None:
			s_use = s_g
		elif self.states[self.s_last] is not None:
			s_use = self.s_last
			#use_mean = True
			print('Extrapolate')
		else:
			while self.states[s_g] is None:
				s_g = random.randint(self.n_states)
		
		if not self.learn:
			noise = 0
		
		q = self.states[s_use].sample_y(x, noise, use_mean)[0]
		
		self.q_g = q
		
		return x, q
	
		
	# PRIVATE
	# Abstract: Find state closest to x
	def get_s(self, x):
		
		dist = np.linalg.norm(self.X - x, axis = 1)
		
		return np.argmin(dist)

	
	# Get neighbors of state.
	def get_neighbors(self, s):
		return self.actions[s]
		
	# Return True if x is in state s. 
	# Faster and more general than get s, and allow for overlap
	def state_contains(self, s, x):
		pass
	
	# Get a random uniform x from within s
	def sample_x(self, s):
		pass
		
	# Abstract: See is two states are neighbors, states are int's here
	def are_neighbours(self, s1, s2):
		pass


#############################################################################

class Sphere(Parent):
	
	def __init__(self, n_states, n_hood_rad = 1.5, diff_max = 1.5):
		
		# Initialization
		# - X 			: Coordinates of states evenly spread out on unit sphere
		# - neigh_dist	: Shortest distance between two states
		
		iterations = 200
		[self.X, self.neigh_dist] = self.get_sphere(n_states, iterations)
		
		self.n_hood_rad = n_hood_rad	# Radius of neighboorhood in unit of shortest
										# distance between two states.
		
		super(Sphere, self).__init__(n_states, state_coord = self.X, diff_max = diff_max)
									
	# END INIT
	
	# PUBLIC
	# Transform X into sperical coordinates
	def cart2sph(self, X):
		
		if X is None:
			return None
		
		if X.ndim == 1:
			theta 	= np.arccos(X[2])
			phi		= np.arctan2(X[1], X[0])
		else:
			theta 	= np.arccos(X[:,2])
			phi		= np.arctan2(X[:,1], X[:,0])
		
		phi = np.mod(phi, 2*np.pi)
		return np.array([theta, phi]).T
		
	# Transform spherical p into carteesian x
	def sph2cart(self, p):
		
		if p is None:
			return None
		
		theta 	= p[0]
		phi		= p[1]
		
		x = np.sin(theta)*np.cos(phi)
		y = np.sin(theta)*np.sin(phi)
		z = np.cos(theta)

		return np.array([x, y, z]).T
	
	# PRIVATE
	def get_sphere(self, n_states, iterations = 500, step_size = 1.):
		
		print('Finding state distribution over Sphere...')
		
		X = np.random.randn(n_states, 3)
		X = self.to_sphere(X)	

		for i in tqdm(range(iterations)):
			
			F, min_dist  = self.get_force(X)
			
			X += step_size*F
			X  = self.to_sphere(X)
			
		return X, min_dist
	
	
	def to_sphere(self, X):
		
		if X.ndim == 1:
			X /= np.linalg.norm(X)
		else:
			for i in range(len(X)):
				X[i] /= np.linalg.norm(X[i])
			
		return X


	def get_force(self, X):
		F = 0.*X
		
		min_dist = 2.
		for i in range(len(X)):
			for j in range(i+1, len(X)):
				
				diff  = X[j] - X[i]
				dist  = np.linalg.norm(diff)
				force = diff/dist**2
				
				F[i] -= force
				F[j] += force
				
				if dist < min_dist:
					min_dist = dist
		
		return F, min_dist
		
	   
	# Return True if x is in state s. 
	# Faster and more general than get s, and allow for overlap
	def state_contains(self, s, x):
		
		if s == -1:
			return False
			
		dist = np.linalg.norm(self.X[s] - x)
		
		return dist < .6*self.neigh_dist	# Allow some overlap
	
	# Get a random uniform x from within s
	def sample_x(self, s):
		
		x_center = self.X[s]
		
		while True:
			x_try = x_center + (np.random.rand(3) - .5)*2*self.neigh_dist
			x_try = self.to_sphere(x_try)
			
			if self.get_s(x_try) == s:
				break
				
		return x_try

		
	# Abstract: See is two states are neighbors
	def are_neighbours(self, s1, s2):
		x1 = self.X[s1]
		x2 = self.X[s2]
		
		dist = np.linalg.norm(x1 - x2)
		
		return dist < self.neigh_dist*self.n_hood_rad
		
	   

##########################################################################
class Cylinder(Sphere):
	
	def __init__(self, n_states, n_hood_rad = 1.5, mem_decay = 0.95, n_neigh = 100, diff_max = 1.5):
		
		# Initialization
		# - X 			: Coordinates of states evenly spread out on unit sphere
		# - neigh_dist	: Shortest distance between two states
		
		super(Cylinder, self).__init__(n_states, n_hood_rad, diff_max = diff_max)
									
	
	# Hack to put everything on cylinder
	def get_sphere(self, n_states, iterations = 0, step_size = 1.):
		
		angles = np.linspace(0,2*np.pi,n_states+1)[:n_states]
		
		X 		 = np.array([np.cos(angles), np.sin(angles), 0*angles]).T
		min_dist = np.linalg.norm(X[0]- X[1])

		return X, min_dist
		
	def sample_x(self, s):
		x_try    = super(Cylinder, self).sample_x(s)
		
		x_try[2] = 0 
		x_try   /= np.linalg.norm(x_try)
		
		return x_try
		
		
################################################################3##############
class Cylinder2(Cylinder):
		
	def __init__(self, n_states, n_hood_rad = 1.5, mem_decay = 0.95, n_neigh = 100, diff_max = 1.5):
		
		# Initialization
		# - X 			: Coordinates of states evenly spread out on unit sphere
		# - neigh_dist	: Shortest distance between two states
		
		super(Cylinder2, self).__init__(n_states, n_hood_rad, diff_max = diff_max)
		
	
	# Hack to put everything on cylinder without belly
	def get_sphere(self, n_states, iterations = 0, step_size = 1.):
		
		a_belly = np.pi/2.
		
		angles 	= np.linspace(a_belly,2*np.pi - a_belly, n_states)
		
		X 		 = np.array([np.cos(angles), np.sin(angles), 0*angles]).T
		min_dist = np.linalg.norm(X[0]- X[1])

		return X, min_dist
		
		

'''
import matplotlib.pyplot as plt  

s = Sphere(50)
X = s.X

n_act = 0
for a in s.actions:
	n_act += len(a)
	print(len(a))
	
print('Mean: ' + str(n_act*1./len(s.actions)))

Y = np.zeros((40, 3))
for i in range(40):
	Y[i] = s.sample_x(0)

plt.figure(0)
plt.scatter(Y[:,1], Y[:,0])
plt.scatter(X[:,1], X[:,0], c=X[:,2], cmap='gray')
plt.axis([-1.1,1.1,-1.1,1.1])

plt.figure(1)
P = s.cart2sph(X)
Q = s.cart2sph(Y)

plt.scatter(Q[:,1], Q[:,0])
plt.scatter(P[:,1], P[:,0], c=X[:,0], cmap='gray')
plt.axis([-np.pi,np.pi,-0,np.pi])

plt.show()
'''
