import numpy as np
import Graphics
import qi

#################### METHODS ############################
def get_Proxies(IP_ADRESS, PORT):

	session = qi.Session()
	try:
		session.connect("tcp://" + IP_ADRESS + ":" + str(PORT))
	except RuntimeError:
		print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
			   "Please check your script arguments. Run with -h option for help.")
		sys.exit(1)

	return session
	
def sph2frame(P):
	if P is None:
		return None
	
	F = 0.*P
	
	if F.ndim == 1:
		F[0] = 1.*P[1]/(2*np.pi)
		F[1] = 1.*P[0]/np.pi
	else:
		F[:,0] = 1.*P[:,1]/(2*np.pi)
		F[:,1] = 1.*P[:,0]/np.pi
	
	return 1 - F
	
def frame2sph(F):
	if F is None:
		return None
	
	P = 0.*F	# Get right length

	P[0] = np.pi*(F[1])
	P[1] = 2.*np.pi*(F[0])

	return P

	
def plot_frame(gph, id_frame, func, f_states, f_pos, f_g = None, f_G = None, \
															state = -1, title = ''):
	state_size = 10
	
	gph.clean_frame(id_frame)
	gph.draw_free_states(id_frame, f_states, 0.*func, state_size)	# Circle around
	
	if state >= 0: # Highlight state
		gph.draw_x_coord(id_frame, f_states[state], Graphics.RED, size = state_size) 
	
	gph.draw_free_states(id_frame, f_states, func, state_size-2)	# Color inside
	gph.draw_x_coord(id_frame, f_pos, Graphics.BLUE, 8)
	
	if f_G is not None:
		gph.draw_cross(id_frame, f_G, Graphics.BLUE)
		
	if f_g is not None:
		gph.draw_x_coord(id_frame, f_g, Graphics.RED, 5)
		
	gph.set_title(id_frame,title)
	
