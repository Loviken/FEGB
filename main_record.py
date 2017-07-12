import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import pygame as pg
import sys
from pygame.locals import *
import cPickle as pickle
import cv2

import Environment  # Forward model
import Intermediary # Approximated inverse model
import Planner2
import Graphics
#import Statistics

# CONSTANTS
#   Meta
iterations   = 10000
seed         = 5

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

filename = 'Resolution_' + str(resolution) + 'x' + str(resolution) + '_Seed' + str(seed) 

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
framesize  = 505
frame_grid = (1,1)
id_env = 0
id_gui = 1

# Define the codec and create VideoWriter object
desktop = False
if desktop:
	fourcc = cv2.VideoWriter_fourcc(*'XVID') # Desktop
else:
	fourcc = cv2.cv.CV_FOURCC(*'XVID') # Laptop
	
video_out = cv2.VideoWriter(filename + '.avi',fourcc, 60.0, (framesize,framesize))

# INIT
#stats = Statistics.Statistics(iterations, sampleFreq_1, sampleFreq_2, N_prec, N_reach, iReach)
d_start = None

np.random.seed(seed)
gph  = Graphics.Basic(framesize, frame_grid, filename)
# Make room fort the plots
gph.draw_matrix(np.zeros((1,1)), id_env, matrix_text = 'Iteration: 0', v_min = -1, v_max = 0)

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
print()
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
		V-= np.min(V)
		V/= np.max(V)
		
		#gph.draw_matrix(V, 0, v_min = -1, v_max = 1)
		
		# nSuccess
		nSuccess = np.log(1 + imd.nSuccess.reshape(imd_res, imd_res))
		#nSuccess-= np.min(nSuccess) # Should be 0
		if np.max(nSuccess) > 0:
			nSuccess/= np.max(nSuccess)

		
		# Get approximation of direct coordinate goal from intermediary
		if pln.training_mode and not no_sample_noise:
			d_goal = imd.get_approximated_direct_state(g_goal)
			
		else:
			d_goal = imd.get_approximated_direct_state(g_goal, True) # No noise
		# Try the approximation and get back the resulting trajectories in g-/d-space
		#gph.draw_g_states(0, g_goal = g_goal)
		if pln.training_mode:
			[D_traj, G_traj] = env.set_motor_output(d_goal, g_goal, background = nSuccess, iterations = epoch, video_out = video_out)
		else:
			[D_traj, G_traj] = env.set_motor_output(d_goal, g_goal, background = nSuccess, iterations = epoch, video_out = video_out, training_mode = pln.training_mode, long_goal = pln.goal_g)
		#gph.draw_g_states(0, g_goal = g_goal, color_goal = (255, 255, 255))

		if pln.training_mode:
			imd.learn(D_traj, G_traj, g_goal)
			#stats.compute_statistics(epoch, env, pln, imd)
			
			if np.mod(epoch,10) == 0:
				print(np.array([100.*epoch/iterations, pln.get_seen_states_proportion(), pln.get_seen_stateactions_proportion()]))
			
			epoch += 1

			
		if epoch == iterations:
			simulation_is_active = False

		
	for event in pg.event.get():
		
		#if user wants to quit
		if event.type == QUIT:
			#end the game and close the window
			
			video_out.release()
			cv2.destroyAllWindows()
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


