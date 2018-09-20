import sys
import numpy as np
import BodyNao
import StateSpace
import FEGB
import Graphics
import Stats
import cPickle as pickle
import time
import copy
from methods import *

	
##################################################################
########################### START ################################
##################################################################

# File management
filenameLOAD = None		# Load from file (None: Start from scratch)
filenameSAVE = None		# Save to file   (None: Don't save)

# About the robot
IP_ADRESS 	 = "192.168.0.111"		# Press Nao's button to learn IP-address
PORT		 = 9559					# Don't change

# FEGB
iterations   = 200
nStates 	 = 9
n_hood_rad	 = 1	# 1: only nearest neighbors as goals, 2: allow nighbors of neighbors,...
max_noise    = 2.	# How much motor babbling to allow. Value of 2 seems to work well empirically
excludeBelly = False #True

# Graphics
framesize  	= 400
frame_grid 	= (2,2)	# (m,n) so that m*n=4

gph = Graphics.Basic(framesize, frame_grid, 'FEGB')
if filenameLOAD is not None:
	pkl_file = open(filenameLOAD + '.pkl', 'rb')
	data  	 = pickle.load(pkl_file)
	fegb  	 = data[0]
	s_space  = fegb.s_space
	epoch 	 = data[1]
	stats 	 = data[2]
	
	iterations += epoch
	
else:
	# 	INIT
	title = 'nStates: ' + str(nStates)

	if excludeBelly:	# Only allow states between back/sides
		s_space = StateSpace.Cylinder2(nStates, n_hood_rad + .5)
	else:				# Allow all rotations (may break shoulders when on belly!)
		s_space = StateSpace.Cylinder(nStates, n_hood_rad + .5)
		
	fegb 	= FEGB.FEGB(s_space, max_noise = max_noise)
	stats 	= Stats.Statistics(nStates, copy.deepcopy(s_space.actions))

	epoch = 0
	

# 	C'EST PARTI!
prox  	= get_Proxies(IP_ADRESS, PORT)
body 	= BodyNao.Body(prox)
body.relax()

while False:
	time.sleep(.2)
	print(np.round(body.get_q(),3))
	
time.sleep(10.)	# Some time to put NAO in a starting position


q = body.get_q()	# The posture = joint angle configuration


# Do a random first movement
q_g = q + .1*np.random.randn(len(q))	
body.set_goal(q_g)	# Prepare robot to attempt a posture change to q_g	
x, q = body.step()	# Make attempt, and observe resulting posture q, and task space position x.
body.relax()
fegb.observe(x, q)  # Improve FEGB using observed new state

q_start = q
x_G = None	# Global (long term) goal, given by user by clicking in graphics
x_g = None	# Next iteration goal, given by planner

while epoch < iterations:
	
	print('\n')
	
	# PLOT GRAPHICS #
	#  The coordinates that are relevant to plot
	f_states  = sph2frame(s_space.cart2sph(s_space.X))
	f_pos	  = sph2frame(s_space.cart2sph(x))
	f_g		  = sph2frame(s_space.cart2sph(x_g))
	f_G		  = sph2frame(s_space.cart2sph(x_G))
	s_i 	  = s_space.s
	s_last	  = s_space.s_last
	
	# Plot the planner's bellieved probability to immediatly reach a state
	P_s = fegb.pln_opti.P[s_last]
	plot_frame(gph, 0, P_s, 		f_states, f_pos, f_g, f_G, s_i, 'Iteration: ' + str(epoch))
	# Plot the planner's state value function
	V = np.log(fegb.pln_opti.V)
	plot_frame(gph, 1, V,		f_states, f_pos, f_g, f_G, s_i, 'V - State value function')
	# Plot number of times each state has been reached
	nSuccess = np.log(.01 + fegb.pln_real.S)
	plot_frame(gph, 2, nSuccess, f_states, f_pos, f_g, f_G, s_i, '# Times reached')
	# Plot believed probability to reach each state
	P_reach = np.max(fegb.pln_real.P, 0)
	plot_frame(gph, 3, P_reach, 	f_states, f_pos, f_g, f_G, s_i, 'Probability to reach')
	
	s_g = fegb.state_goal
	#gph.draw_x_coord(0, f_states[s_g], Graphics.BLUE, size = 10) 
	gph.update()
	
	# NEXT POSTURE
	fegb.set_goal(x_G)			# if x_G=None, use intrinsic motivation
	x_g,q_g = fegb.get_goal()
	
	body.set_goal(q_g)
	x, q = body.step()

	fegb.observe(x, q)	# Improve FEGB using observed new state
	
	# CHECK FOR INPUT FROM USER
	f_G = gph.check_events()
	if f_G is not None:
		print(f_G)
		if min(f_G) < 0 or max(f_G) >= 1:
			x_G = None
		else:
			x_G = s_space.sph2cart(frame2sph(1.-f_G))
			
			print(sph2frame(s_space.cart2sph(x_G)))
			
	
	seen_s = np.sum(fegb.pln_real.S > 0)
	print('Iteration: ' + str(epoch) + '\tSeen states: ' + str(seen_s))
	
	# statistics
	s_f = s_space.s
	x_f = x
	stats.observe(s_i, s_g, s_f, x_g, x_f)

	
	if False: # failed:
		print('Failed to reach posture.')
		print(np.max(np.abs(q_g - q)))
	
	# ITERATE
	epoch += 1
	
body.relax()

# Save model?
if filenameSAVE is not None:
	filename = filenameSAVE + '.pkl'
	y_or_n 	 = raw_input('Save robot as \'' + filename + '\'?  (y/n) :')

	if y_or_n is 'y':				
		output 	 = open(filename, 'wb')
		pickle.dump([fegb, iterations, stats], output)
		output.close()
