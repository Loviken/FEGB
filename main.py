import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import pygame as pg
import sys
from pygame.locals import *
import cPickle as pickle

import Environment  # Forward model
import Intermediary # Approximated inverse model
import Planner2
import Graphics
#import Statistics

# CONSTANTS
#   Meta
iterations   = 10000
seed         = 4

filename = '100 DoF Arm - Seed ' + str(seed)

#  Statistiscs
'''
sampleFreq_1	= 10		# How often to check most stats
sampleFreq_2	= 100		# How often to check reach
N_prec			= 10**2		# Sample size precission
N_reach			= 10		# Sample size reach goal
iReach			= 10		# nActions to reach goal
'''

# Environment
dof 		 = 100
armLength 	 = 1.0			# Arm is in middle, side is 1
nSteps 		 = 100			# how many steps to take each transition
resolution   = 6			# make nxn grid
imd_res      = resolution	# Another resolution for intermediary?
wall_setting = 8
random_start = False

filename = 'Resolution: ' + str(resolution) + 'x' + str(resolution) + ', Seed: ' + str(seed) 

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

# Graphics
framesize  = 300
frame_grid = (3,2)
id_gui = 0
id_env = 1
buttons = ('Play/Pause', 'Save', 'Quit')

# INIT
#stats = Statistics.Statistics(iterations, sampleFreq_1, sampleFreq_2, N_prec, N_reach, iReach)
d_start = None

np.random.seed(seed)
gph  = Graphics.Basic(framesize, frame_grid, filename, id_gui = id_gui, buttons = buttons)
# Make room fort the plots
gph.draw_matrix(np.zeros((1,1)), id_env, matrix_text = str(dof) + ' DoF Arm', v_min = -1, v_max = 0)
gph.draw_matrix(np.zeros((1,1)), 3, matrix_text = 'V: State-value function')
gph.draw_matrix(np.zeros((1,1)), 4, matrix_text = 'Next step probability', v_min = 0, v_max = 1)
gph.draw_matrix(np.zeros((1,1)), 5, matrix_text = 'nSuccess', v_min = 0)
gph.draw_matrix(np.zeros((1,1)), 2, matrix_text = 'Success Rate')

env  = Environment.Arm(gph, id_frame = id_env, d_start = d_start, dof = dof, wall_setting=wall_setting, random_start = random_start, nSteps = nSteps, armLength = armLength)

d_start = env.get_direct_state()
g_start = env.global_state

imd  = Intermediary.StateInverse(env.dim, imd_res, d_start, max_noise = max_noise, forget_threshold = forget_threshold, collision_avoid = collision_avoid)
imd.learn(d_start.reshape(1,-1), g_start.reshape(1,-1), g_start)

pln  = Planner2.Curiosity(resolution, g_start, start_prob = start_prob, decay = decay, P_0 = P_0, gamma = gamma, Q_thr = Q_thr, random_act = random_act, returnIfFail = returnIfFail)

epoch = 0

# Train the model
simulation_is_active = True

print(env.global_state)
time.sleep(1)
while True:
	
	if simulation_is_active or not pln.training_mode:
		
		# PLAN NEXT GLOBAL GOAL
		g_current = env.global_state
		g_goal    = pln.get_next_goal(g_current)
		
		# Update graphics
		# V: State-value function
		V = pln.value_func.reshape(resolution, resolution)
		V = np.log(V)
		gph.draw_matrix(V, 3)
		gph.draw_g_states(3, g_current, g_goal)
		
		# Success probability
		s_i = pln.get_state(g_current)
		prob_reach_state = pln.trans_prob[s_i].diagonal() 
						
		prob_matrix = prob_reach_state.reshape(resolution, resolution)
		gph.draw_matrix(prob_matrix, 4, v_min = 0, v_max = 1)
		gph.draw_g_states(4, g_current, g_goal)
		
		# nSuccess
		nSuccess = np.log(1 + imd.nSuccess.reshape(imd_res, imd_res))
		gph.draw_matrix(nSuccess, 5, v_min = 0)
		gph.draw_g_states(5, g_current, g_goal)
		
		# Success Rate
		success_rate = imd.nSuccess/(imd.nTries + 1)
		success_rate = success_rate.reshape(imd_res, imd_res)
		gph.draw_matrix(success_rate, 2, v_min = 0, v_max = 1)
		
		# Get approximation of direct coordinate goal from intermediary
		if pln.training_mode and not no_sample_noise:
			d_goal = imd.get_approximated_direct_state(g_goal)
			
		else:
			d_goal = imd.get_approximated_direct_state(g_goal, True) # No noise
		# Try the approximation and get back the resulting trajectories in g-/d-space
		#gph.draw_g_states(0, g_goal = g_goal)
		#t_start = time.time()
		[D_traj, G_traj] = env.set_motor_output(d_goal, g_goal)
		#gph.draw_g_states(0, g_goal = g_goal, color_goal = (255, 255, 255))
		#print('time: ' + str(time.time() - t_start))

		if pln.training_mode:
			imd.learn(D_traj, G_traj, g_goal)
			#stats.compute_statistics(epoch, env, pln, imd)
			
			if np.mod(epoch,10) == 0:
				print(np.array([100.*epoch/iterations, pln.get_seen_states_proportion(), pln.get_seen_stateactions_proportion()]))
			
			epoch += 1
		
		'''	
		if np.mod(epoch+1,20) == 0:
			while True:
				x_i = np.random.rand(2) # Approximate start
				q_i = imd.get_approximated_direct_state(x_i, no_noise = True)
				x_i = env.get_G_from_D(q_i)	# Real start
					
				angles  = 2.*(q_i - 0.5)*env.aMax 
				coord_q = env.getCoordinates(angles)
					
				if env.isLegal(coord_q):
					env.angles = angles
					env.coord  = coord_q
					pln.get_next_goal(x_i)
					imd.learn([q_i], [x_i], x_i, actually_learn = False)
						
					break
		'''
			
		if epoch == iterations:
			simulation_is_active = False
			
			# Save stats for later use in notebook etc...
			'''
			output 	 = open('stats_' + filename + '.pkl', 'wb')
			pickle.dump(stats, output)		
			output.close()
			
			s_seen 	= stats.states_seen
			sa_seen = stats.state_act_seen
			prec 	= stats.precision
			t 		= np.array(range(iterations/sampleFreq_1))*sampleFreq_1
			print(len(t))
			
			plt.plot(t,s_seen)
			plt.show()
			plt.plot(t,sa_seen)
			plt.show()
			plt.plot(t,prec)
			plt.show()
			'''

		
	for event in pg.event.get():
		
		#if user wants to quit
		if event.type == QUIT:
			#end the game and close the window
			pg.quit()
			sys.exit()
			
		elif event.type == MOUSEBUTTONDOWN:
			window_target = pg.mouse.get_pos()
			[frame_id, frame_pos] = gph.get_frame_and_pos(window_target)
			
			if frame_id != id_gui:
				g_coord = gph.get_g_from_frame(frame_pos)
				
				if g_coord.min() >= 0 and g_coord.max() < 1 :
					pln.training_mode = False
					pln.set_user_goal(g_coord)
					
					print('Training \'off\' ')
				else:
					pln.training_mode = True
					print('Training \'on\' ')
					
			else:
				button_id = gph.get_button(frame_pos)
				
				# Play/Pause
				if button_id == 0:
					simulation_is_active = not simulation_is_active
				
				# Save
				elif button_id == 1:
					filename = raw_input('Save as: ')
					output 	 = open(filename + '.pkl', 'wb')		#open(str(resolution) + 'x' + str(resolution) + '_' + str(i) + 'iter_' + filename + '.pkl', 'wb')

					env.remove_graphics()
				
					data = [imd, pln, env]
					pickle.dump(data, output)
					
					output.close()
					
					env.add_graphics(gph, id_env)
					
				# Quit
				elif button_id == len(buttons) - 1:
					#end the game and close the window
					pg.quit()
					sys.exit()
