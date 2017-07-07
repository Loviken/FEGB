import numpy as np
import cv2
import pygame as pg
import copy
import time

class Parent(object):
	
	def __init__(self, gp = np.array([0.6,0.6])):
		
		self.global_state = gp		# this value can store a position pressed by the user, and thus
										# extracted.
		self.dim = [1,1]			# Dimensions in [G, D]

	def do(self, action):	# Do the action and return resulting (g,d)-trajectories
		return 0
		
	def get_direct_state(self):
		pass

	def set_motor_output(self, d_goal):
		pass


################################# IMAGE ENVIRONMENT ###################################
class Image(Parent):
	
	def __init__(self, filename = 'chessboard.jpg'):
		super(Image, self).__init__()
		self.image = cv2.imread(filename)
		self.height, self.width, _ = self.image.shape
		self.updateSteps = 100	# Data points to collect for every movement.
		self.dim = [2,3]			# Dimensions in [G, D]
		
	def get_direct_state(self, g_state = []):
		if g_state == []:
			g_state = self.global_state
		
		x_int = np.int32(g_state[0]*self.width)
		y_int = np.int32(g_state[1]*self.height)
		
		clrs = np.float32(self.image[y_int,x_int])/255
		
		return clrs
		
		
	def set_motor_output(self, d_goal):
		# This function will be a little bit silly for the image, since we lack a mechanism for
		# how the "change of color value" would move us around in g-space. Thus I will just say
		# that the motor command will move it to a random g-position in N steps.
		
		g_ini = self.global_state
		g_end = np.clip(g_ini + np.random.randn(2)*0.1, 0, 0.999)	# I will ignore d_goal for this toy-example
		
		G_traj = np.zeros((self.updateSteps, 2))
		D_traj = np.zeros((self.updateSteps, 3))
		
		G_traj[:,0] = np.linspace(g_ini[0], g_end[0], self.updateSteps + 1)[1:]
		G_traj[:,1] = np.linspace(g_ini[1], g_end[1], self.updateSteps + 1)[1:]
		
		for i in range(self.updateSteps):
			D_traj[i] = self.get_direct_state(G_traj[i])
			
		self.global_state = g_end
		
		return D_traj, G_traj




################################# 100 DoF ARM ###################################

import cv2

#CONSTANTS
WHITE	= (255,255,255)
BLACK	= (0,0,0)
RED     = (255,50,50)
BLUE    = (50,50,255)

LINE_WIDTH = 3



class Arm(Parent):
		
	# dof   : How many DoF the arm has
	# nTurns : How many circles the arm should be able to make with max angles
	def __init__(self, graphics = 0, random_start = False, armLength = 1.5, dof = 100, id_frame = 1, d_start = None, wall_setting = -1, nSteps = 100):
		super(Arm, self).__init__()
		
		# THE AGENT
		self.dof 	= dof
		self.dim 	= [2,dof]
		self.segLen = armLength/dof		# Length of an arm segment
		self.aMax   = np.pi
		aStart 		= 1.*4*np.pi/dof	# This is just so that it can unfold nicely
		self.angles = np.linspace(aStart,aStart/4,dof)
		self.coord0 = np.array([0.5, 0.5])
		self.coord  = self.getCoordinates(self.angles)
		self.g_goal0  = np.array([0.5,0.5])
		
		# GRAPHICS
		self.graphics  = graphics							# Use this canvas for plotting etc
		if graphics != 0:
			self.displace = np.array(graphics.get_canvas_displacement(id_frame))	# This is where the plotting starts
			self.wall_displace = np.append(self.displace , np.array([0,0]))
			self.screen    = graphics.screen
			self.frameside = graphics.frameside*graphics.canvas_prop
		
		# THE UPDATES
		self.firstRound  = True
		self.updateSteps = nSteps
		
		# THE SURROUNDING
		cW = 0.2  # corridor width
		wW = 0.02 # wall thickness
		
									
		self.walls 	= np.array([[0,0,0,0]])
		
		choice = wall_setting # Add walls
		if choice == 0:	
			self.walls 	= np.array([[.2, .2, .2, .2]])
		elif choice == 1:
			self.walls 	= np.array([[ 1 - cW - wW, 2*cW, wW, 2*cW],
									[ 0.  , 4*cW, 1 - cW, wW]])
									#[ cW, 1 - cW - wW, 1 - cW, wW],
									#[ cW, 2*cW, wW, 1 - 3*cW]])
		elif choice == 2:
			cW = 1./6
			self.walls 	= np.array([[   cW,   1.5*cW, wW, 1 - 3*cW],
									[1-wW-cW, 1.5*cW, wW, 1 - 3*cW]])
		elif choice == 3:
			self.walls 	= np.array([[1.5*cW,   cW, 1 - 3*cW, wW],
									[1.5*cW, 1-wW-cW, 1 - 3*cW, wW],
									[   cW,   1.5*cW, wW, 1 - 3*cW],
									[1-wW-cW, 1.5*cW, wW, 1 - 3*cW]])
		elif choice == 4: # One u-shaped wall above
			self.walls 	= np.array([[ 1 - cW - wW, cW, wW, cW],
									[ cW, cW, 1 - 2*cW, wW],
									[ cW, cW, wW, cW]])
									
		elif choice == 5: # One u-shaped wall bellow
			self.walls 	= np.array([[          cW, 1 - cW - wW, 1 - 2*cW, wW],
									[ 1 - cW - wW,    1 - 2*cW,       wW, cW],
									[          cW,    1 - 2*cW,       wW, cW]])
									
		elif choice == 6: # Simple wall bellow
			self.walls 	= np.array([[2*cW, 1 - cW - wW, cW, wW]])
			
		elif choice == 7: # Long wall above
			self.walls 	= np.array([[ cW, 1-cW, 1 - 2*cW, wW]])
			
		elif choice == 8: # cross
			self.walls 	= np.array([[ .5-.5*wW,        0, wW, cW], 
									[ .5-.5*wW,     1-cW, wW, cW],
									[ 0       , .5-.5*wW, cW, wW],
									[ 1-cW    , .5-.5*wW, cW, wW]])
									
		# UPDATE RELATED
		self.angleStart = copy.deepcopy(self.angles)
		self.angleGoal 	= np.zeros(dof)	# start by unfolding
		
		if False: #True: # random start
			if d_start == None:
					
				while True:
					a_try = np.random.randn(dof).clip(-self.aMax, self.aMax)
					
					if self.isLegal(self.getCoordinates(a_try)):
						break
					
				self.angles = a_try
					
				#self.update(self.g_goal0)
			else:
				self.angles = 2.*(d_start - 0.5)*self.aMax
			
		else:
			self.set_motor_output(self.angleGoal + .5, self.g_goal0) #self.update(self.g_goal0)
			
		self.firstRound = False
		
	
	def add_graphics(self, graphics, id_frame):
		self.graphics		= graphics 
		self.displace 		= np.array(graphics.get_canvas_displacement(id_frame))	# This is where the plotting starts
		self.wall_displace 	= np.append(self.displace , np.array([0,0]))
		self.screen    		= graphics.screen
		self.frameside 		= graphics.frameside*graphics.canvas_prop
		
	def remove_graphics(self):
		self.graphics		= None
		self.displace 		= None
		self.wall_displace 	= None
		self.screen    		= None
		self.frameside 		= None
		
	def get_direct_state(self):
		return self.angles / (self.aMax*2) + 0.5	# Change span from [-a,a] to [0,1]
		
	def set_motor_output(self, d_goal, g_goal, background = None, iterations = 0, video_out = None, training_mode = True, long_goal = np.zeros(2)):
		self.setGoal((2.*d_goal - 1)*self.aMax)

		[D_traj, G_traj]  = self.update(g_goal, background, iterations, video_out, training_mode, long_goal)
		self.g_goal0 = g_goal
		
		self.global_state = G_traj[-1,:]
				
		return D_traj/(self.aMax*2) + 0.5, G_traj 	# Change span from [-a,a] to [0,1]
		
		
	# Make sure no point is illegal	
	# add constraint to see if kinetic and potential energy doesn't increase too much
	def isLegal(self, coord_d):
		
		# Within bounds?
		if np.min(coord_d) < 0 or np.max(coord_d) >= 1:
			return False
			
		# If dof is too low points will be too far away for collision test
		segLen = self.segLen
		
		if len(coord_d) < 101:
			new_coord = np.zeros((101, 2))
			new_coord[:,0] = np.interp(np.linspace(0,1,101),\
										np.linspace(0,1,len(coord_d)), coord_d[:,0])
			new_coord[:,1] = np.interp(np.linspace(0,1,101),\
										np.linspace(0,1,len(coord_d)), coord_d[:,1])
										
			segLen *= self.dof*1./101
			
			coord_d = new_coord
		
		# Would it go through walls?
		walls = self.walls
		
		for i in range(len(walls)):
			for j in range(len(coord_d)):
				x  = coord_d[j,0]
				y  = coord_d[j,1]
				x1 = walls[i,0]
				x2 = x1 + walls[i,2]
				y1 = walls[i,1]
				y2 = y1 + walls[i,3]
				
				if x1 < x and x < x2 and y1 < y and y < y2:
					#print('Wall collision!')
					return False
		
		# Would it go through itself?
		#  I will do this simpler by drawing a circle around every segment and see if two segments
		#  intersect
		radius = segLen*0.95
		
		# Sort by x for faster clasification
		coord_d = coord_d[ coord_d[:,0].argsort() ]

		for i in range(len(coord_d)):
			for j in range(i+1, len(coord_d)):
				dx = np.abs(coord_d[i,0] - coord_d[j,0])
				dy = np.abs(coord_d[i,1] - coord_d[j,1])
				
				#print([i,j,dx, radius])
				if dx > radius:
					break
				
				if dy + dx < radius:
					return False
					
				#if dx**2 + dy**2 < radius**2:
				#	return False
		
		return True
		
	# Transform angles of arm to coordinates
	def getCoordinates(self, angles):
		coordX = np.zeros(self.dof + 1) + self.coord0[0]
		coordY = np.zeros(self.dof + 1) + self.coord0[1]
		tmpAng = np.pi/2	#start angle
		
		for i in range(self.dof):
			tmpAng += angles[i]
			
			coordX[i+1] = coordX[i] + self.segLen*np.cos(tmpAng)
			coordY[i+1] = coordY[i] + self.segLen*np.sin(tmpAng)
			
		return np.array([coordX, coordY]).T
		
	# Get a global coordinate from a direct
	def get_G_from_D(self, d):
		angles = 2.*(d - 0.5)*self.aMax
		coord  = self.getCoordinates(angles)
		
		return coord[-1,:]
		
		
	def setGoal(self, angle_goal):
		self.angleStart = self.angles # This do what?
		self.angleGoal = angle_goal
		

	def update(self, g_goal, background = None , iterations = None, video_out = None, training_mode = True, long_goal = np.zeros(2)):
		# This will update and redraw the agent
		
		data_direct = np.zeros((self.updateSteps + 1, self.dof))
		data_global = np.zeros((self.updateSteps + 1, 2)) # Outputspace is 2, for now...
		
		data_direct[0,:] = self.angleStart
		data_global[0,:] = self.getCoordinates(self.angleStart)[-1,:]
		
		g_dist_start = np.linalg.norm(g_goal - data_global[0,:])
		
		# Update graphics
		if self.graphics != 0:
			if background == None:
				goal_pos = self.displace + np.int32(self.g_goal0*self.frameside)
				pg.draw.circle(self.screen, WHITE, goal_pos, 4)
				
				self.drawWalls()
			
			goal_pos = self.displace + np.int32(g_goal*self.frameside)
		
		#stats
		
		for i in range(1, self.updateSteps + 1): 	# only update if changed
			frac = (1.*i)/self.updateSteps #np.sin(0.5*np.pi*i/self.updateSteps)	# prop to goal
			tmpAngles = (1. - frac)*self.angleStart + frac*self.angleGoal
			tmpCoord  = self.getCoordinates(tmpAngles)

			if self.isLegal(tmpCoord):
				if self.graphics != 0:
					if background == None:
						self.drawAgent(self.coord, WHITE) # Paint over old
					else:
						self.graphics.screen.fill(WHITE)
						self.graphics.draw_matrix(background, 0, v_min = -1, v_max = 1, matrix_text = 'Iteration: ' + str(iterations))
						self.drawWalls()
				
				self.angles = tmpAngles
				self.coord  = tmpCoord
				
				data_direct[i,:] = tmpAngles
				data_global[i,:] = tmpCoord[-1,:]
				
				if self.graphics != 0:
					self.drawAgent(self.coord, BLACK)
					pg.draw.circle(self.screen, RED, goal_pos, 4)
					if not training_mode:
						long_pos = self.displace + np.int32(long_goal*self.frameside)
						o1 = 5*np.array([1,1])
						o2 = 5*np.array([-1,1])
						
						pg.draw.line(self.screen, BLUE, long_pos-o1, long_pos+o1, 4)
						pg.draw.line(self.screen, BLUE, long_pos-o2, long_pos+o2, 4)
					
					pg.display.update()
					
					if video_out != None and (iterations <= 50 or not training_mode):
						frame = np.transpose(pg.surfarray.array3d(self.graphics.screen), axes=(1, 0, 2))
						video_out.write(frame)
					
				g_dist = np.linalg.norm(data_global[i,:] - data_global[0,:])
				if g_dist >= g_dist_start and not self.firstRound:
					return [data_direct[0:i,:], data_global[0:i,:]]
			else:
				self.global_state = data_global[i,:]
				
				if video_out != None and (iterations > 50 and training_mode):
					frame = np.transpose(pg.surfarray.array3d(self.graphics.screen), axes=(1, 0, 2))
					video_out.write(frame)
				
				return [data_direct[0:i,:], data_global[0:i,:]]
		
		self.global_state = data_global[-1,:]
		if video_out != None and (iterations > 50 and training_mode):
			frame = np.transpose(pg.surfarray.array3d(self.graphics.screen), axes=(1, 0, 2))
			video_out.write(frame)

		return [data_direct, data_global]
		
	def drawAgent(self, coord, col):
		# this will scale and draw a line between the coordinates
		# col is the color of the lines
		
		pg.draw.lines(self.screen, col, False, self.displace + coord*self.frameside, LINE_WIDTH)
		
	def drawWalls(self):
		walls = self.walls
		
		for i in range(len(walls)):
			rect = self.wall_displace + walls[i]*self.frameside
			pg.draw.rect(self.screen, BLACK, rect)

'''
agent = Arm()

i = 0
while i < 200:
	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			quit()
	
 #event.type == pg.MOUSEBUTTONDOWN:
	goal = agent.angles + 0.2*(2*np.random.rand(agent.dof)-1)
	goal = goal.clip(- agent.aMax, agent.aMax)
	agent.setGoal(goal)
		
	a,b = agent.update()
	#print(a)
	i += 1
	print(i)
'''
