import numpy as np
import Environment  # Forward model
import Intermediary # Approximated inverse model
import Planner2

import copy
from   tqdm import tqdm

class Statistics:
	
	def __init__(self, nIterations, sampleFreq_1, sampleFreq_2, N_inv_error, N_reach_error, iReach):
		self.nIterations 		= nIterations
		self.sampleFreq_1		= sampleFreq_1		# How often to sample
		self.sampleFreq_2		= sampleFreq_2		# How often to do reach test (comp-expensive)
		self.N_inv_error		= N_inv_error		# How many computationally "cheap" samples to collect
		self.N_reach_error		= N_reach_error		# How many reach-samples to collect (Expensive!)
		self.iReach				= iReach			# How many steps to allow when reaching for something.

		self.states_seen 	= np.zeros(nIterations/sampleFreq_1)
		self.state_act_seen	= np.zeros(nIterations/sampleFreq_1)
		
		self.inv_error		= np.zeros((nIterations/sampleFreq_1, N_inv_error))
		
		if sampleFreq_2 > 0:
			self.reach_error	= np.zeros((nIterations/sampleFreq_2, N_reach_error))	# This is more time-consuming


	def compute_statistics(self, i, env, pln, imd):
		
		if np.mod(i, self.sampleFreq_1) == 0:
			print("Collecting data type 1")
			
			self.states_seen[i/self.sampleFreq_1 - 1] 		= pln.get_seen_states_proportion()
			self.state_act_seen[i/self.sampleFreq_1 - 1]	= pln.get_seen_stateactions_proportion()
			
			error = np.zeros(self.N_inv_error)
			for j in range(self.N_inv_error):
				
				# Find a position for which an inverse model is defined
				while True:	
					x_Goal = np.random.rand(2)
					state  = imd.get_state(x_Goal)
					
					if imd.nSuccess[state] != 0:
						break
				
				q_approx 	= imd.get_approximated_direct_state(x_Goal, no_noise = True)
				x_real		= env.get_G_from_D(q_approx)
				
				error[j] = np.linalg.norm(x_Goal - x_real)
			
			self.inv_error[i/self.sampleFreq_1 - 1] = error
			
			print(np.array([100.*i/self.nIterations, pln.get_seen_states_proportion(), pln.get_seen_stateactions_proportion()]))
			print('Inverse model error: ' + str(np.mean(error)))
			
		if self.sampleFreq_2 > 0 and np.mod(i, self.sampleFreq_2) == 0:
			print("Collecting data type 2")
			
			# Must copy seens intermediary would add new states.
			env_tmp = copy.deepcopy(env)
			pln_tmp = copy.deepcopy(pln)
			imd_tmp = copy.deepcopy(imd)
			
			pln_tmp.training_mode = False
			env_tmp.graphics = 0
			
			error = np.zeros(self.N_reach_error)
			
			for j in tqdm(range(self.N_reach_error)):
				while True:
					x_i = np.random.rand(2) # Approximate start
					q_i = imd_tmp.get_approximated_direct_state(x_i, no_noise = True)
					x_i = env_tmp.get_G_from_D(q_i)	# Real start
					
					angles  = 2.*(q_i - 0.5)*env_tmp.aMax 
					coord_q = env_tmp.getCoordinates(angles)
					
					if env_tmp.isLegal(coord_q):
						env_tmp.angles = angles
						env_tmp.coord  = coord_q
						pln_tmp.get_next_goal(x_i)
						imd_tmp.learn([q_i], [x_i], x_i, actually_learn = False)
						
						break
				
				x_Goal = np.random.rand(2)
				
				pln_tmp.set_user_goal(x_Goal)
				
				# Try to go there
				min_inv_error = 1.
				for k in range(self.iReach):
					
					# plan next step
					x_current = env_tmp.global_state
					x_goal    = pln_tmp.get_next_goal(x_current)
					
					# Get approximation of direct coordinate goal from intermediary
					q_goal = imd_tmp.get_approximated_direct_state(x_goal, no_noise = True)
					
					# Apply motor-signal
					[Q_traj, X_traj] = env_tmp.set_motor_output(q_goal, x_goal)
					imd_tmp.learn(Q_traj, X_traj, x_goal, actually_learn = False)

					# The result
					x_real = env_tmp.global_state
					dist   = np.linalg.norm(x_Goal - x_real)
					
					if dist < min_inv_error:
						min_inv_error = dist
				
				#if i > 0:
				#	print(min_inv_error)
					
				error[j] = min_inv_error
				
			self.reach_error[i/self.sampleFreq_2 - 1] = error
			
			print('Reach error: ' + str(np.mean(error)))
