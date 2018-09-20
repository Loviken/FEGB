import numpy as np
		
# Observe that the special cases state '-1' is treated separately and is not included in V, Q, P, R, etc...
class Curious:
	
	def __init__(self, n_states, decay = 0.9, P_0 = 0.2, gamma = 0.95, Q_thr = 10**(-6)):
		
		# Attributes
		self.P 		= np.zeros((n_states, n_states))	# Approximated prob: s -> s'
		self.R_cur	= .00001*np.random.rand(n_states, n_states) + 1		# Curiosity reward:  s -> s'
		self.R_exp	= np.zeros((n_states, n_states))	# Exploit reward:    s -> s'

		# Parameters
		self.decay 	= decay		# Update rate of probability-avarages
		self.P_0	= P_0		# Successful transitions become more interesting is prob < P_0
		self.gamma  = gamma		# Discounted future reward in MDP
		self.Q_thr  = Q_thr 	# Value iteration threshold

		# Value iteration
		# - These are rather placeholders
		self.V		= np.zeros(n_states)				# State-value function
		self.Q		= np.zeros((n_states, n_states))	# State-Action-value function

		# Statistics
		#  S[s]: Times state s has been visited
		self.S  = np.zeros(n_states)
		#  SA[s1,s2]: Times transition s->a has been attempted
		self.SA = np.zeros((n_states,n_states))


	# Observe a transition s_i->s_f, where s_g was the goal
	def observe(self, s_i, s_g, s_f):
		
		p0 		= self.P_0					# Don't loose interest if p =< p0
		alpha 	= self.decay				# Decay of probabilities
		beta 	= p0/(alpha*p0 + 1 - alpha)	# Decay of curiosity-reward
		
		if s_i != -1:
			#self.P[s_i, s_g] = 1.
			
			self.S[s_i]		  += 1
			self.SA[s_i, s_g] += 1	# Tries are recorded to see what option are tried
			
			self.P[s_i, s_g] *= alpha
			
			if s_f == s_g:
				self.P[s_i, s_f] 	 += (1 - alpha)		# Increase P in case of success
				
				self.R_cur[s_i, s_f] *= beta			# Decrease cur-rew in case of success
				
			elif s_f != -1 and self.P[s_i, s_f] > 0:	# Since s_i -> s_f worked, such a transition is simulated.
				# This wouldn't work for pessimistic planner, but protects from illegal transitions
				self.P[s_i, s_f] *= alpha
				self.P[s_i, s_f] += (1 - alpha)
				
				self.R_cur[s_i, s_f] *= beta			# Decrease cur-rew in case of success

		
	# Return the action/goal-state with highest Q-value, given state s, 
	#  and possible goal states A = [s1,s2,...]
	def get_goal(self, s, A, s_G = None):
		
		# special case, off manifold of solutions
		if s == -1:
			i_g = np.random.randint(len(A))	#index goal
			
		else:
			_,Q = self.get_VQ(s_G)
			
			sQ  = Q[s,A]		# Expected return for different choices
			i_g = np.argmax(sQ)

			#print('Actions: ' + str(A) + '\nP: ' + str(self.P[s,A]) + '\nR: ' + \
			#		str(self.R_cur[s,A]) + '\nQ: ' + str(Q[s,A]) + '\nChoice: ' + str(A[i_g]))

		return A[i_g]

	
	# Get state-value matrix V and state-action-value matrix Q.
	# This is done using state value iteration.
	def get_VQ(self, s_G = None):

		# Parameters
		max_diff = self.Q_thr
		gamma 	 = self.gamma
		
		# Main matrices
		self.Q *= 0.
		self.V *= 0.
		
		P = self.P
		if s_G is None: # Curiosity
			R = self.R_cur
			
		else:	# Exploitation
			self.R_exp 			*= 0.	# Forget old
			self.R_exp[s_G, s_G] = 1.
			
			R = self.R_exp

		# VALUE ITERATION
		iterations = 0	# This is only used if it doesn't converge (Shouldn't happen)
		while True:
			Q_new  = P*(R + gamma*self.V)
			diff   = np.max(np.abs(Q_new - self.Q))
			
			self.Q = Q_new
			self.V = np.max(self.Q, axis=1)

			if diff < max_diff:
				break
			
			iterations += 1 
			if iterations > 10**5:
				print('V and Q couldn\'t converge!')
				break
				
		return 1.*self.V, 1.*self.Q
