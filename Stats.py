import numpy as np
import copy

# Statistics

class Statistics:
	def __init__(self, n_states, actions):
		
		# Constants
		self.n_states  	= n_states
		self.actions	= actions	# Lists the possible actions for every state
		self.legal_sa	= []		# Legal state actions
		
		for s in range(n_states):
			for a in actions[s]:
				sa = self.get_sa(s,a)
				self.legal_sa.append(sa)
				
		self.legal_sa = np.array(self.legal_sa)
		
		# This section keeps track on the current state
		# and is thus updated.
		self.S_success   = np.zeros(n_states)
		self.S_attempts  = np.zeros(n_states)
		self.SA_success  = np.zeros(n_states**2)
		self.SA_attempts = np.zeros(n_states**2)
		self.SA_prec	 = [None]*n_states**2	# List of precisions for every SA

		# This section saves the values of every iteration
		self.success_vs_fail = []	# True = success, ...
		self.chosen_SA		 = []	# indicates the choosen SA (Added after 1'st collection)
		self.all_S_success 	 = [copy.deepcopy(self.S_success)]
		self.all_S_attempts	 = [copy.deepcopy(self.S_attempts)]
		self.all_SA_success  = [copy.deepcopy(self.SA_success)]
		self.all_SA_attempts = [copy.deepcopy(self.SA_attempts)]
		self.all_SA_prec 	 = [copy.deepcopy(self.SA_prec)]
		

	def observe(self, s_i, s_g, s_f, x_g, x_f):
		
		if s_g != -1:	# There was an actual goal
			self.S_attempts[s_g] += 1
			
			if s_i != -1: # Started on manifold
				sa = self.get_sa(s_i, s_g)
				
				self.chosen_SA.append(sa)
				self.SA_attempts[sa] += 1
				
				if s_f == s_g:	# Success
					self.success_vs_fail.append(True)
					
					self.S_success[s_g] += 1
					self.SA_success[sa] += 1
					
				else:	# Failure
					self.success_vs_fail.append(False)
					
				if self.SA_prec[sa] is None:
					self.SA_prec[sa] = []
					
				prec = np.linalg.norm(x_g - x_f)
				self.SA_prec[sa].append(prec)
		
			# Added code, wasn't used for 1'st collection
			self.all_S_success.append(copy.deepcopy(self.S_success))
			self.all_S_attempts.append(copy.deepcopy(self.S_attempts))
			self.all_SA_success.append(copy.deepcopy(self.SA_success))
			self.all_SA_attempts.append(copy.deepcopy(self.SA_attempts))
			self.all_SA_prec.append(copy.deepcopy(self.SA_prec))
		

	# Get state action of moving from s1 to s2
	def get_sa(self, s1, s2):
		return s1*self.n_states + s2
		
################################################################################

class Statistics2:
	def __init__(self, n_states, actions):
		
		# Constants
		self.n_states  	= n_states
		self.actions	= actions	# Lists the possible actions for every state
		self.legal_sa	= []		# Legal state actions
		
		for s in range(n_states):
			for a in actions[s]:
				sa = self.get_sa(s,a)
				self.legal_sa.append(sa)
				
		self.legal_sa = np.array(self.legal_sa)
		
		# Records to keep
		self.S_i = []
		self.S_g = []
		self.S_f = []
		
		self.SA_try 	= []	# State-action tried
		self.SA_outcome = []	# Whether s-a was successful
		self.SA_error	= []	# Continous distance between x_g and x_f
		
		
	def observe(self, s_i, s_g, s_f, x_g, x_f):
		
		self.S_i.append(s_i)
		self.S_g.append(s_g)
		self.S_f.append(s_f)
		
		sa = self.get_sa(s_i, s_g)
		
		self.SA_try.append(sa)
		self.SA_outcome.append(s_g == s_f)
		self.SA_error.append(np.linalg.norm(x_g - x_f))
		
	# Get state action of moving from s1 to s2
	def get_sa(self, s1, s2):
		return s1*self.n_states + s2
		
		
		
		
		
		
		
