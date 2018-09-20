import Planner
import StateSpace
import numpy as np

class FEGB:

	def __init__(self, s_space, forget_threshold = .2, max_noise = 1.0):
	
		# Attributes
		self.state_goal 	= -1	# Last state goal
		self.forget_thresh	= forget_threshold
		self.s_space  		= s_space
		self.max_noise		= max_noise
		
		self.x_G	  = None			# A goal position in task space that can be set from a user.
		
		self.pln_real = Planner.Curious(self.s_space.n_states)	# Realistic planner
		self.pln_opti = Planner.Curious(self.s_space.n_states)	# Optimistic planner
		
		for s in range(self.s_space.n_states):
			#self.pln_real.P[s,s] = 1.	# Should be able to stay in state
			self.pln_opti.S[s] = 1.
			
			for a in self.s_space.get_neighbors(s):
				# Start assumption is 100% success for optimistic planner
				self.pln_opti.P[s,a]  = 1.
				self.pln_opti.SA[s,a] = 1.


	# PUBLIC METHODS
	
	# Observe a trajectory.
	# - When exploiting the model to reach a goal x_G it will only observe
	def observe(self, x, q):
		s_i = self.s_space.s	# Initial state
		s_g = self.state_goal	# Goal state
		
		self.s_space.observe(x,q)		
		
		s_f = self.s_space.s	# Final state
		print(str(s_i) + ' -> ' + str(s_f) + ' (' + str(s_g) + ')')
		
		if self.x_G is None:	# Train
			self.pln_real.observe(s_i, s_g, s_f)
			self.pln_opti.observe(s_i, s_g, s_f)
			
			# Forget if odds are too low
			#if False: # Use if random explore
			if s_i != -1 and s_g != s_f:
				if self.pln_opti.P[s_i, s_g] < self.forget_thresh:
					self.s_space.states[s_g]  = None
					self.pln_opti.P[s_i, s_g] = 1.
					self.pln_real.S[s_g]	  = 0.
					print('State (' + str(s_g) + ') was forgotten...')
						
			

	# To clear goal, simply don't give an argument.
	def set_goal(self, x_G = None):
		self.x_G = x_G
		self.s_space.learn = (x_G is None)
	
	
	# Return a task and motor goal
	def get_goal(self, use_mean = False):
		s = self.s_space.s			# Initial state
		
		As = self.s_space.get_neighbors(s)	# Action space of current region
		
		#print([s, self.s_space.s_last])
		
		if s == -1:
			if np.random.rand() < .5:
				s_g = self.s_space.s_last # Try to go back to manifold

			else:
				reg  = self.s_space.region
				As   = self.s_space.get_neighbors(reg)
			
				s_g  = As[np.random.randint(len(As))]
			
			noise = self.max_noise/(self.pln_real.S[s_g] + 1.)
			
		elif self.x_G is None:
			# Exploration
			s_g = self.pln_opti.get_goal(s, As)
			#self.pln_real.get_goal(s, As)	# update V and Q	
			
			noise  = .1 + self.max_noise/(self.pln_real.S[s_g] + 1.)
			noise *= 1. - self.pln_opti.P[s, s_g]	# Desperation prop to confidence
						
		else:
			# Exploitation
			noise = 0
			
			s_G = self.s_space.get_s(self.x_G)			# Long term goal
			#s_g = self.pln_real.get_goal(s, As, s_G)	# Next step goal
			s_g = self.pln_opti.get_goal(s, As, s_G)	# Next step goal
		
		noise  = max(noise, .05)				# Always use some noise
		
		print('Noise level: ' + str(noise))
		self.state_goal = s_g
		
		x,q = self.s_space.get_goal(s_g, self.x_G, noise = noise, use_mean = use_mean) # Remember expl noise if x_G != None
		
		return x,q
		
		

		
