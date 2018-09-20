# Interface to Nao

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import qi

# Constants
DOF 		= 26

#MAX_RAD 	= 1.	# 1. (pi) is max?
STIFFNESS 	= 1. # .8 # Unless v5: <= .5
ARM_STIFF	= STIFFNESS # .2 # 
SPEED	 	= .4 #.3
MOVE_TIME 	= 2.	# How much time to give each motion
#Q_TOLERANCE = 1. #.5	# How close to goal posture to avoid fail set [.2, .5] resonable

RANGES = np.array( [[-2.0857, 2.0857],		#0	HeadYaw
					[-0.6720, 0.5149],		#1	HeadPitch
					[-2.0857, 2.0857],		#2	LShoulderPitch
					[-0.3142, 1.3265],		#3	LShoulderRoll
					[-2.0857, 2.0857],		#4	LElbowYaw
					[-1.5446,-0.0349],		#5	LElbowRoll
					[-1.8238, 1.8238],		#6	LWristYaw
					[-1, 1],				#7	LHand
					[-1.145303, 0.740810],	#8	LHipYawPitch
					[-0.379472, 0.790477],	#9	LHipRoll
					[-1.535889, 0.484090],	#10 LHipPitch
					[-0.092346, 2.112528],	#11 LKneePitch  (?)
					[-1.189516, 0.922747],	#12 LAnklePitch (?)
					[-0.39, 	0.68],		#13 LAnkleRoll
					[-1.145303, 0.740810],	#14	RHipYawPitch (Same servo as 8)
					[-0.790477, 0.379472],	#15 RHipRoll
					[-1.535889, 0.484090],  #16 RHipPitch
					[-0.103083, 2.120198],	#17 RKneePitch
					[-1.186448, 0.932056],	#18 RAnklePitch
					[-0.68, 	0.39],		#19 RAnkleRoll
					[-2.0857, 2.0857],  	#20 RShoulderPitch
					[-1.3265, 0.3142],		#21 RShoulderRoll
					[-2.0857, 2.0857],		#22 RElbowYaw
					[ 0.0349, 1.5446],		#23 RElbowRoll
					[-1.8238, 1.8238],		#24 RWristYaw
					[-1, 1]])				#25 RHand

# From: http://doc.aldebaran.com/1-14/family/robots/joints_robot.html#robot-joints-v4-head-joints


class Body:
	def __init__(self, proxy):
		
		self.session = proxy
		self.motion_service = proxy.service("ALMotion")
		self.memory_service = proxy.service("ALMemory")

		# Deactivate fall reflex (make sure deactivation is enabled on "Former robot web page")
		self.motion_service.setFallManagerEnabled(False)
		
		self.q_goal = np.array([0.]*DOF)
		
		
	# Set q as the next goal, where q is a np-array
	def set_goal(self, q):
		self.q_goal	= q
			
	
	# Return end position (x,q) - use regulator
	def step(self):
		
		self.motion_service.setStiffnesses("Body", STIFFNESS)
		self.motion_service.setStiffnesses("Arms", ARM_STIFF)
		
		x = self.get_x()
		q = np.array(self.motion_service.getAngles('Body', True))
		
		# Go to new goal
		self.motion_service.setAngles("Body", self.q_goal.tolist(), SPEED)
		
		# Regulator
		P = .4 #.4
		I = .05  #0.05
		D = 0.
		
		I_error 	= 0.*self.q_goal	# Start with no accumulated error
		q_motor 	= 1.*self.q_goal	# motor signal
		q_motor_old = 1.*self.q_goal
		# Regulator - end
		
		t_start = time.time()
		p = .1	# Use to create running mean for x (to reduce noise)
		
		t = time.time() - t_start
		while t < MOVE_TIME:
			# Last values
			x_new = self.get_x()
			q_new = np.array(self.motion_service.getAngles('Body', True))
			q_old = 1.*q
			
			x = p*x_new + (1-p)*x
			q = p*q_new + (1-p)*q # q_new # 
			
			#dQ.append(q - self.q_goal)

			t_old 	= t
			t 		= time.time() - t_start

			#Regulate#
			if t > MOVE_TIME*.3:
				P_error	 = self.q_goal - q			# Direct response
			
				I_error += P_error
				D_error  = (q - q_old)/(t-t_old)
				
				q_motor_old = q_motor
				q_motor 	= self.q_goal + P*P_error + I*I_error + D*D_error
				
				q_motor = q_motor.clip(-np.pi, np.pi)
				self.motion_service.setAngles("Body", q_motor.tolist(), SPEED)
			
		if True: # cylinder
			x[2] = 0

		x /= np.linalg.norm(x)
		
		if False: # Check how well goal posture was reached
			dQ = np.array(dQ)
			plt.plot(dQ)
			plt.show()
		
		return x, q
		
	#############################################################
	# Return end position (x,q) - use regulator
	def step2(self):
		
		self.motion_service.setStiffnesses("Body", STIFFNESS)
		self.motion_service.setStiffnesses("Arms", ARM_STIFF)
		
		x  = self.get_x()
		q  = np.array(self.motion_service.getAngles('Body', True))
		q0 = 1.*q
		qR = np.ones(q.shape)	# What motors needs full power?
		
		dQ = []	# Control how well the agent is reaching goal postures
		
		p = .1	# Use to create running mean for x (to reduce noise)
		t_start = time.time()
		while time.time() - t_start < MOVE_TIME:
			q_goal = self.q_goal + 10.*np.sign(self.q_goal-q0)*qR
			q_goal = q_goal.clip(-np.pi, np.pi)
			self.motion_service.setAngles("Body", q_goal.tolist(), SPEED)
			
			time.sleep(.01)
			# Last values
			x_new = self.get_x()
			q_new = np.array(self.motion_service.getAngles('Body', True))
			
			x = p*x_new + (1-p)*x
			q = p*q_new + (1-p)*q # q_new # 
			
			dQ.append(q - self.q_goal)

			q_pass = (np.sign(self.q_goal - q0) != np.sign(self.q_goal - q))
			
			qR[q_pass] *= 0
			
			
		if True: # cylinder
			x[2] = 0

		x /= np.linalg.norm(x)
		
		if True: # Check how well goal posture was reached
			dQ = np.array(dQ)
			plt.plot(dQ)
			plt.show()
			
		diff   = np.abs(q - self.q_goal)
		failed = np.max(diff) > Q_TOLERANCE
		
		if failed:
			print('Failed with diff: ' + str(np.max(diff)))
		else:	
			print('Success with diff: ' + str(np.max(diff)))
		
		return x, q, failed
	######################################################################
		
	def get_q(self):
		return np.array(self.motion_service.getAngles('Body', True))
		
		
	def get_random_q(self):
		q_mean = np.mean(RANGES, axis = 1)
		
		q = q_mean + (np.random.rand(DOF) - .5)*(RANGES[:,1] - RANGES[:,0])
		
		return q
		
		
	def relax(self):
		self.motion_service.setStiffnesses("Body", 0.)
		
		
	# PRIVATE #
	# Returns x in raw form
	def get_x(self):
		memory_service = self.memory_service

		# Get the Accelerometers Values
		AccX = memory_service.getData("Device/SubDeviceList/InertialSensor/AccX/Sensor/Value")
		AccY = memory_service.getData("Device/SubDeviceList/InertialSensor/AccY/Sensor/Value")
		AccZ = memory_service.getData("Device/SubDeviceList/InertialSensor/AccZ/Sensor/Value")

		x = np.array([AccX, AccY, AccZ])
		
		return x

