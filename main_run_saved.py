import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pygame as pg
import sys
from pygame.locals import *
import cPickle as pickle

import Environment  # Forward model
import Intermediary # Approximated inverse model
import Planner2
import Graphics

filename = 'Room8_res6x6_100dof_arm'

# Load
pkl_file = open(filename + '.pkl', 'rb')
data     = pickle.load(pkl_file)
iRun  	 = 0 #

imd     = data[0][iRun]
pln     = data[1][iRun]
env		= data[2][iRun]

# Now let's add some graphics to it
framesize  = 300
frame_grid = (3,2)
id_gui = 0
id_env = 1
buttons = ('Play/Pause', 'Save', 'Quit')

# Initialize
gph  = Graphics.Basic(framesize, frame_grid, filename, id_gui = id_gui, buttons = buttons)

env.add_graphics(gph, id_env)

pln.training_mode = False

# Random start pose
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


# Make room fort the plots
gph.draw_matrix(np.zeros((1,1)), id_env, matrix_text = str(env.dof) + ' DoF Arm', v_min = -1, v_max = 0)
gph.draw_matrix(np.zeros((1,1)), 3, matrix_text = 'V: State-value function')
gph.draw_matrix(np.zeros((1,1)), 4, matrix_text = 'Next step probability', v_min = 0, v_max = 1)
gph.draw_matrix(np.zeros((1,1)), 5, matrix_text = 'nSuccess', v_min = 0)
gph.draw_matrix(np.zeros((1,1)), 2, matrix_text = 'Success Rate')

# Train the model
simulation_is_active = False

while True:
	if simulation_is_active:
		# PLAN NEXT GLOBAL GOAL
		g_current = env.global_state
		g_goal    = pln.get_next_goal(g_current)
		d_goal 	  = imd.get_approximated_direct_state(g_goal, True) # No noise since test
			
		# DO IT
		[D_traj, G_traj] = env.set_motor_output(d_goal, g_goal)
		imd.learn(D_traj, G_traj, g_goal, actually_learn = False)
		
		# GRAPHICS
		# V: State-value function
		V = pln.value_func.reshape(pln.resolution, pln.resolution)
		V = np.log(V)
		gph.draw_matrix(V, 3)
		gph.draw_g_states(3, g_current, g_goal)
		
		# Success probability
		s_i = pln.get_state(g_current)
		prob_reach_state = pln.trans_prob[s_i].diagonal()  #imd.get_prob_matrix()
		prob_matrix = prob_reach_state.reshape(pln.resolution, pln.resolution)
		gph.draw_matrix(prob_matrix, 4, v_min = 0, v_max = 1)
		gph.draw_g_states(4, g_current, g_goal)
		
		# nSuccess
		nSuccess = np.log(1 + imd.nSuccess.reshape(imd.resolution, imd.resolution))
		gph.draw_matrix(nSuccess, 5, v_min = 0)
		gph.draw_g_states(5, g_current, g_goal)
		
		# Success Rate
		success_rate = imd.nSuccess/(imd.nTries + 1)
		success_rate = success_rate.reshape(imd.resolution, imd.resolution)
		gph.draw_matrix(success_rate, 2, v_min = 0, v_max = 1)

	
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
					pln.set_user_goal(g_coord)
					
					print('New goal: ' + str(g_coord))
					
			else:
				button_id = gph.get_button(frame_pos)
				
				# Play/Pause
				if button_id == 0:
					simulation_is_active = not simulation_is_active
				
				# Save
				elif button_id == 1:
					filename = raw_input('Save as: ')
					output 	 = open(filename + '.pkl', 'wb')		#open(str(resolution) + 'x' + str(resolution) + '_' + str(i) + 'iter_' + filename + '.pkl', 'wb')
					
					data = [imd, pln, env]
					
					pickle.dump(data, output)
					
					output.close()
					
				# Quit
				elif button_id == len(buttons) - 1:
					#end the game and close the window
					pg.quit()
					sys.exit()
