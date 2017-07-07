import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import copy
import Inverse

######################## STATE INVERSE ###########################
# Creates an inverse model for every state. It only saves dataopints
# that originates from the right distribution, which can be seen in
# that the agent ended up where it wanted to end up, or that it ended
# up somewhere where earlier data was missing.

class StateInverse:
	
	def __init__(self, dim, resolution, d_start, mem_decay = 0.95, n_neigh = 100, max_noise = .1, forget_threshold = .1, collision_avoid = True):
		self.dimG = dim[0]	# Dimensionality of global coordinates
		self.dimD = dim[1]	# Dimensionality of direct coordinates
		self.resolution  = resolution
		self.nStates	 = resolution**2
		self.on_manifold = True
		
		self.inv_model 			= [None]*self.nStates	# The inverse models
		self.n_neigh   			= n_neigh	# How many neighbours to take into acount
		self.alpha	   			= 1e-5		# Expected noise-level (Should be neiglible, just to avoid singular cov)
		self.mem_decay 			= mem_decay	# How much used memories fade
		self.max_noise 			= max_noise	# How much noise to put on samples
		self.forget_threshold	= forget_threshold # When to give up a state.
		self.collision_avoid	= collision_avoid
		
		self.nTries    = np.zeros(self.nStates)
		self.nSuccess  = np.zeros(self.nStates)
		self.nSamples  = np.zeros(self.nStates)	# How many samples I have of a state
		
		self.use_stuck	 = False
		self.stuck 		 = False	# Is the agent stuck?
		self.times_stuck = 0		# Frustration increases when stuck, to allow it to get out
		self.max_stuck   = 10		# Panic if stuck too long
		
		self.last_d = d_start
		self.last_g = np.zeros(self.dimG)

			
	def get_approximated_direct_state(self, g_g, no_noise = False):
		s_g = self.get_state(g_g)
		
		if self.nSuccess[s_g] == 0:
			s_use = self.get_state(self.last_g)
			noise = self.max_noise
		else:
			s_use = s_g
			
			if no_noise:
				noise = 0
			else:
				noise = self.max_noise/self.nSuccess[s_use]

		d_state = self.inv_model[s_use].sample_y(g_g, noise, keep_mem = no_noise)[0] + noise**2*np.random.randn(self.dimD) #+ noise**2*np.random.rand()*np.random.randn(self.dimD)
					
		return d_state.clip(0,1)
		

	def learn(self, D_traj, G_traj, g_g, actually_learn = True):
		s_i = self.get_state(G_traj[0])
		s_g = self.get_state(g_g)
		s_f = self.get_state(G_traj[-1])
		
		if len(D_traj) == 1 :
			self.stuck = True
		else:
			self.stuck = False
		
		if actually_learn:
			if self.collision_avoid:
				precision = 1. - 1./len(D_traj)	# The bigger batch, the better data since we didn't get stuck.
			else:
				precision = 1.
			
			# Did we succeed?
			if s_g == s_f:
				self.nSuccess[s_g] += 1
			elif self.on_manifold:
				self.nTries[s_i]   += 1
				
			if self.nSuccess[s_g] > 0: # and self.on_manifold:
				self.nTries[s_g] += 1
				
			# Should it be forgotten?
			if self.nSuccess[s_g] > 0 and 1.*self.nSuccess[s_g]/(self.nTries[s_g] + .00001) < self.forget_threshold: #This is only change from testing
				print('State ' + str(s_g) + ' was forgotten.')
				self.inv_model[s_g] = Inverse.Model(self.n_neigh, self.alpha, self.mem_decay)
				
				self.nTries[s_g]   = 0
				self.nSuccess[s_g] = 0
				self.nSamples[s_g] = 0

		
		self.last_d = D_traj[-1]
		try_last_g  = G_traj[-1]
		
		# If not learning we must remember last defined state.
		if actually_learn or self.nSuccess[self.get_state(try_last_g)] > 0:
			self.last_g = try_last_g
		
		# Go through all the seen d_vectors
		for i in range(len(D_traj)):
			
			# # We must avoid confusing a starting position in the right state but on the wrong
			# # branch to the goal position
			#if s_i == s_g and not self.on_manifold:
			#	break
			
			s_tmp = self.get_state(G_traj[i])
			
			update_d = False
			
			# Are we still on manifold (of learned solutions)?
			if self.on_manifold:
				
				if s_tmp != s_i and s_tmp != s_g:
					self.on_manifold = False
			
			# Are we back on the manifold? Don't learn is s_i=s_g since danger!
			elif s_tmp == s_g and s_i != s_g:
				self.on_manifold = True
				
			
			# Update vector if on manifold...
			if self.on_manifold:
				update_d = True
				
			# ...or if a new state is reached.
			if self.nSamples[s_tmp] == 0 and actually_learn:
				print('State ' + str(s_tmp) + ' was found.')
				self.inv_model[s_tmp] = Inverse.Model(self.n_neigh, self.alpha, self.mem_decay)
				
				self.nTries[s_tmp]   = 1
				self.nSuccess[s_tmp] = 1
				self.nSamples[s_tmp] = 0
				
				update_d = True
			
				
			# Update d if apropriate
			if update_d and actually_learn:
				G = G_traj[i].reshape(1,-1)
				D = D_traj[i].reshape(1,-1)
				
				self.inv_model[s_tmp].fit(G, D, precision)
				self.nSamples[s_tmp] += 1


	def get_state(self, g_coord):
		discrete = np.int32(np.floor((g_coord*self.resolution*0.9999)))
		return discrete[0]*self.resolution + discrete[1]
		
		
	def get_prob_matrix(self):
		state_prob = 1. - self.nSuccess/(self.nTries + 0.01) 
		r = self.resolution
		
		return state_prob.reshape(r,r)
		
