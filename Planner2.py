import numpy as np
		

class Curiosity:
	# This is Curiosity based Reinforcement Learning. NOPE, it was though...
	
	def __init__(self, resolution, g_coord, use_state_reward = False, start_prob = 1., decay = 0.8, P_0 = 0.2, gamma = 0.95, Q_thr = 10**(-4), random_act = False, returnIfFail = False):
		self.resolution  = resolution
		self.n_states    = resolution**2
		self.n_st_ac	 = 0						# Number of state actions
		self.state_start = self.get_state(g_coord)	# Last state
		self.state_goal  = 0		# Last state goal
		self.training_mode = True
		self.use_state_reward = use_state_reward		# Use reach-state-at-all reward
		
		self.failed_last  = False
		self.returnIfFail = returnIfFail
		
		self.decay 	= decay
		self.P_0	= P_0
		self.gamma  = gamma
		self.Q_thr  = Q_thr 			# Value iteration threshold
		
		self.random_act = random_act	# Pick actions randomly
		
		# User defined
		self.goal_g = g_coord
		self.goal_s = 0
				
		n_states = self.n_states 
		# RECORDED TRANSITIONS: T(s_i, s_g, s_f)
		# - The number of times an attempted transition s_i -> s_g led to s_f.
		self.trans_seen     = np.zeros((n_states, n_states, n_states))
		self.state_act_seen = np.zeros((n_states, n_states)) # number of attempts s_i,s_g
		self.state_seen     = np.zeros(n_states)
		
		# PROBABILITY DISTRIBUTION: P(s_i, s_g, s_f)
		# - The probability that an attempt to go from s_i to s_g will lead to s_f.
		self.trans_prob = np.zeros((n_states, n_states, n_states))
		# - The probability to end up in a state when trying to get there
		self.state_prob = start_prob*np.ones(n_states)
		
		# REWARD FUNCTION: R(s_i, s_g)
		# - The reward of trying to go to s_g while in s_i
		self.trans_reward = np.ones((n_states, n_states))
		self.state_reward = np.ones(n_states)
		self.user_reward  = np.ones((n_states, n_states))
		
		# Q-FUNCTION: Q(s_i, s_g)
		# - The expected future reward of aiming for s_g in state s_i
		self.Q_func = np.zeros((n_states, n_states))
		
		# VALUE FUNCTION: V(s_i)
		# - The expected future reward of being in state s_i
		self.value_func = np.zeros((n_states))
		
		# MANIFOLD OF SOLUTIONS
		self.on_manifold = True	# Are we where we are by skill (True) or accident (False)
		
		# INITIATE THE WHOLE THING
		for s_i in range(n_states):
			neigh = self.get_possible_goal_states(s_i)
			self.n_st_ac += len(neigh)
			
			for s_g in neigh:
				self.trans_prob[s_i, s_g, s_g] = start_prob
		
		self.update_V_and_Q()
		
		print('Initiate Planner - Done')


	# A method that updates the system as a new observation is made.
	#
	#   s_i : Initial state
	#   s_g : Goal state
	#   s_f : Final State
	def observe(self, s_i, s_g, s_f):
		
		if self.on_manifold:
			self.trans_seen[s_i, s_g, s_f] += 1
			self.state_act_seen[s_i, s_g] += 1
			self.state_seen[s_f] += 1
		
			decay 	= self.decay	# Update rate of probabilities
			p0 		= self.P_0		# Don't loose interest if p =< p0
			
			# Update transitional probabilities
			self.trans_prob[s_i, s_g, :]   *= decay
			self.trans_prob[s_i, s_g, s_f] += (1 - decay)*1
			
			if s_g == s_f:
				# Decay reward for every visit.
				
				beta = p0/(decay*p0 + 1 - decay)
				self.trans_reward[s_i, s_f] *= beta
			
		# Are we on manifold?		
		self.on_manifold = (s_f == s_g or s_f == s_i)
		

	# Some methods to convert between state numbers g_positions
	def get_state(self, g_coord):
		discrete = np.int32(np.floor((g_coord*self.resolution*0.9999)))
		
		return discrete[0]*self.resolution + discrete[1]


	def get_g_coord(self, state, displacement = 0.5):
		x = np.int32(state / self.resolution)
		y = np.mod(state, self.resolution)
		
		disp = displacement/self.resolution
		g_coord = np.array([x,y])*1./self.resolution + disp
		
		return g_coord
		
	
	# A method that will return a list of possible goal states
	# - This defines the available actions in a state.
	# - For now it is assumed that boundries are non-looping
	def get_possible_goal_states(self, state):
		g_coord  = self.get_g_coord(state)
		distance = 1./self.resolution
		
		neighbors = [state]	# All von Neumann
		for i in range(8):
			angle = 2.*i*np.pi/8
			displacement = distance*np.array([np.cos(angle), np.sin(angle)])
			
			g_neighbor = g_coord + displacement
			
			if 0 <= min(g_neighbor) and max(g_neighbor) < 1:
				neighbors.append(self.get_state(g_neighbor))
			
		return np.array(neighbors)


	#  If the user sets a goal state
	def set_user_goal(self, g_coord):
		goal_state = self.get_state(g_coord)
		
		self.user_reward *= 0
		self.user_reward[goal_state, goal_state] = 1	# go to goal state and stay there
		
		self.goal_g = g_coord
		self.goal_s = goal_state
		
		self.update_V_and_Q()

	
	def update_V_and_Q(self):
		delta_threshold = self.Q_thr
		gamma = self.gamma
		
		tmp = 0
		
		#self.value_func *= 0.
		iterations = 0
		while True:
			delta_max = 0.	# Biggest delta seen in loop
			
			for s_i in range(self.n_states):
				possible_goals = self.get_possible_goal_states(s_i)
				
				for s_g in possible_goals:
					# For every state-action...
					
					# Q(s,a) = R(s,a) + gamma * SUM_i[P(s,a,i)*V(i)]
					if self.training_mode:
						reward = self.trans_reward[s_i, s_g]
							
					else:
						reward = self.user_reward[s_i, s_g]
					
					Q_new = self.trans_prob[s_i, s_g, s_g]*(reward + gamma*self.value_func[s_g])
								
					diff = abs(Q_new - self.Q_func[s_i, s_g])
					delta_max = max(diff, delta_max)
					
					# Update Q-function
					self.Q_func[s_i, s_g] = Q_new
					
				# Update Value-function
				self.value_func[s_i] = max(self.Q_func[s_i, possible_goals])

			tmp += 1
			
			if delta_max < delta_threshold:
				break
			
			iterations += 1 
			if iterations > 10**3:
				print('V and Q couldn\'t converge!')
				break
				
		#self.Q_func = .1*self.Q_func + .9*old_Q
	
	
	# ANALYTICS
	def get_seen_states_proportion(self):
		n_seen_states = np.sum(np.int8(self.state_seen != 0))
		return 1.*n_seen_states/self.n_states
		
		
	def get_seen_stateactions_proportion(self):
		n_seen_state_actions = np.sum(np.int8(self.state_act_seen != 0))
		
		return 1.*n_seen_state_actions/self.n_st_ac
		

	# This method gets a new state and can learn from it by remembering its previous state
	#   and goal state. It then updates Q and V befor returning a new goal state, in its
	#   gloobal coordinates.
	def get_next_goal(self, g_coord):
		state_new = self.get_state(g_coord)
		
		if state_new != self.state_goal:
			
			self.failed_last = True
				
		else:
			self.failed_last = False
			
					
		if self.training_mode:
			self.observe(self.state_start, self.state_goal, state_new)
			self.update_V_and_Q()
		
		if self.failed_last and state_new == self.state_start and self.returnIfFail:
			self.state_goal = self.state_start
		else:
			self.state_goal  = np.argmax(self.Q_func[state_new])
			self.state_start = state_new
		
		# Use this for random actions
		if self.random_act and self.training_mode:
			neigh = self.get_possible_goal_states(state_new)
			index = np.random.randint(0, len(neigh))
			self.state_goal = neigh[index]
		
		g_now  = g_coord
		
		# If in execution mode in vicinity of human set goal
		if not self.training_mode and self.state_goal == self.goal_s:
			g_goal = self.goal_g
			
		# If on path to human set goal
		else:
			g_goal = self.get_g_coord( self.state_goal, displacement=np.random.rand(2) )
			#disp = (.1*np.random.randn(2)+.5).clip(0,1)
			#g_goal = self.get_g_coord( self.state_goal, disp )
			
				
		return g_goal
