import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cPickle as pickle
import time

import Environment  # Forward model
import Intermediary # Approximated inverse model
import Planner2
import Statistics

# CONSTANTS
#   Meta
iterations   = 50000
seed         = 0
n_runs		 = 1		# How many samples to collect

# Environment
dof 		 = 100
armLength 	 = 1.0			# Arm is in middle, side is 1
nSteps 		 = 100			# how many steps to take each transition
resolution   = 10			# make nxn grid
imd_res      = resolution	# Another resolution for intermediary?
wall_setting = 8
random_start = False

# Intermediary
max_noise 			= .15		# The variance of the samples
forget_threshold 	= .1		# How low successrate before giving up on state.
collision_avoid 	= False		# Punnish trajectories that end abruptly
no_sample_noise		= False 	# Only when training 
mem_decay 			= .95

# Planner
random_act  = False
returnIfFail= True		# If transition fails and still in starting state, change pos in s_i
start_prob 	= 1.		# Assume this probability of success from the start
decay 		= 0.8		# Decay old probabilities by this amount
P_0 		= 0.3		# Increase action reward if initial probability is bellow P_0
gamma 		= 0.95		# Discount of future reward
Q_thr 		= 10**(-4)	# Max update-error in value-iteration

#  Statistiscs
sampleFreq_1	= iterations/200	# How often to check most stats
sampleFreq_2	= iterations/50		# How often to check reach
N_inv_error		= 200				# Sample size inv.model error
N_reach_error	= 100				# Sample size reach goal error
iReach			= 3*resolution		# nActions to reach goal

filename = 'Room' + str(wall_setting) + '_res' + str(resolution) + 'x' + str(resolution) + \
			'_'   + str(dof) + 'dof'

# INITIALIZE
statistics	 = [None]*n_runs
#environment	 = [None]*n_runs
#intermediary = [None]*n_runs
#planner		 = [None]*n_runs

np.random.seed(seed)
for epoch in range(n_runs):
	
	d_start 	 = None

	stats = Statistics.Statistics(iterations, sampleFreq_1, sampleFreq_2, N_inv_error, N_reach_error, iReach)
	env   = Environment.Arm(random_start = random_start, dof = dof, wall_setting = wall_setting, \
							nSteps = nSteps, armLength = armLength)

	d_start = env.get_direct_state()
	g_start = env.global_state

	imd  = Intermediary.StateInverse(env.dim, imd_res, d_start, max_noise = max_noise, forget_threshold = forget_threshold, collision_avoid = collision_avoid, mem_decay = mem_decay)
	imd.learn(d_start.reshape(1,-1), g_start.reshape(1,-1), g_start)

	pln  = Planner2.Curiosity(resolution, g_start, start_prob = start_prob, decay = decay, P_0 = P_0, gamma = gamma, Q_thr = Q_thr, random_act = random_act, returnIfFail = returnIfFail)

	# Train the model
	#stats.compute_statistics(0, env, pln, imd)
	for i in range(iterations):
		
		if np.mod(i+1, sampleFreq_2) == 0:
			print('\nEPOCH: ' + str(epoch+1) + '/' + str(n_runs) + ' - ' + filename)
		
		# PLAN NEXT GLOBAL GOAL
		g_current = env.global_state
		g_goal    = pln.get_next_goal(g_current)

		# Get approximation of direct coordinate goal from intermediary
		d_goal = imd.get_approximated_direct_state(g_goal, no_sample_noise)

		[D_traj, G_traj] = env.set_motor_output(d_goal, g_goal)
		
		imd.learn(D_traj, G_traj, g_goal)
		stats.compute_statistics(i + 1, env, pln, imd)
		
	statistics[epoch] 	= stats
	#environment[epoch]	= env
	#intermediary[epoch] = imd
	#planner[epoch]		= pln
		
			
# Save stats for later use in notebook etc...
output 	 = open(filename + '_stats.pkl', 'wb')
pickle.dump(statistics, output)		
output.close()

# Save arm?
choice = raw_input('Save arm ' + filename + '? (y/n): ')
if choice == 'y':
	output 	 = open(filename + '_arm.pkl', 'wb')
	
	#data = [intermediary, planner, environment]
	data = [imd, pln, env]
	
	pickle.dump(data, output)
	
	output.close()
