import numpy as np
import matplotlib as mpl
from math import sin, cos, atan2, inf, pi
from ir_sim.world import RobotBase

class RobotTwoWheel(RobotBase):

	robot_type = 'custom'
	appearance = 'circle'
	state_dim  = (3, 1) # the state dimension, x, y, theta(heading direction),
	vel_dim = (2, 1)    # the angular velocity of right and left wheel
	goal_dim = (3, 1)	# the goal dimension, x, y, theta 
	position_dim = (2, 1) # the position dimension, x, y

	def __init__(self, id, state, vel = np.zeros((2, 1)), goal=np.zeros((3, 1)), 
				 step_time = 0.01, **kwargs):
		self.shape  = kwargs.get('shape', [4.6, 1.6, 3, 1.6]) # Only for rectangle shape
		self.radius = kwargs.get('radius', 0.25)
		self.radius_wheel = kwargs.get('radius_wheel', 1)
		self.control_mode = kwargs.get('control_mode', 'auto')
		self.noise_mode   = kwargs.get('noise_mode', 'none')
		self.noise_amplitude = kwargs.get('noise_amplitude', np.c_[[0.005, 0.005, 0.001]])
		super(RobotTwoWheel, self).__init__(id, state, vel, goal, step_time, **kwargs)

	def dynamics(self, state, vel, **kwargs):
		r"""
		Choose dynamics function based on different noise mode.
		"""
		if self.control_mode == 'keyboard':
			vel = self.keyboard_to_angular(vel)

		if self.noise_mode == 'none':
			return self.dynamics_without_noise(state, vel, **kwargs)
		elif self.noise_mode == 'linear':
			return self.dynamics_with_linear_noise(state, vel, **kwargs)
		else:
			return self.dynamics_with_nonlinear_noise(state, vel, **kwargs)


	def dynamics_without_noise(self, state, vel, **kwargs):
		r"""
		Question 1
		The dynamics of two-wheeled robot, be careful with the defined direction of the robot.
		
		Some parameters that you may use:
		@param dt: delta time
		@param r:  wheel radius
		@param l:  distancce from wheel to center
		@param vel:   2*1 matrix, the angular velocity of the right and left wheels, [omega1, omega2]
		@param state: 3*1 matrix, the state dimension, [x, y, theta (heading direction)]
		
		Return:
		@param next_state: 3*1 matrix, same as state
		"""
		dt = self.step_time
		r  = self.radius_wheel
		l  = self.radius
		
		"*** YOUR CODE STARTS HERE ***"

		next_state = np.zeros((1,3))
		#print(next_state)

		r_ICR = ((-2*l*vel[0]) / (vel[0]-vel[1])) + l
		#print(r_ICR)

		omega = ((vel[0]-vel[1])*r) / (2*l) 
		#print(omega)

		matrix_A = np.array([ [cos(omega*dt) , -sin(omega*dt), 0],  
							  [sin(omega*dt) , cos(omega*dt) , 0],
							  [0             , 0             , 1]])

		x_tminus1 = np.array([ -r_ICR*sin(state[2]), 
							   r_ICR*cos(state[2]), 
							   state[2]])

		matrix_B = np.array([state[0] + r_ICR*sin(state[2]),
							 state[1] - r_ICR*cos(state[2]),
							 omega * dt])

		immed = np.matmul(matrix_A, x_tminus1)
		#print(immed)
		output = immed + matrix_B
		#print(output)

		next_state = output

		"*** YOUR CODE ENDS HERE ***"
		return next_state


	def dynamics_with_linear_noise(self, state, vel, **kwargs):
		r"""
		Question 2(a)
		The dynamics of two-wheeled robot, be careful with the defined direction.
		
		Some parameters that you may use:
		@param dt: delta time
		@param r:  wheel radius
		@param l:  distancce from wheel to center
		@param vel:   2*1 matrix, the angular velocity of the right and left wheels, [omega1, omega2]
		@param state: 3*1 matrix, the state dimension, [x, y, theta (heading direction)]
		@param R:     3*1 matrix, noise amplitude of the additive Gaussian disturbances 
						for the state, [R_x, R_y, R_theta]
		
		Return:
		@param next_state: 3*1 matrix, same as state
		"""
		dt = self.step_time
		r  = self.radius_wheel
		l  = self.radius
		R  = self.noise_amplitude
		
		"*** YOUR CODE STARTS HERE ***"

		next_state = np.zeros((1,3))

		if vel[0] == vel[1]:
			#omega1 = omega2 --- the robot is walking on straight line, r_ICR is infinite, omega is 0

			output = np.array([state[0] + vel[0] * 1 * cos(state[2]) * dt, 
							   state[1] + vel[0] * 1 * sin(state[2]) * dt,
							   state[2] ])

			#add noise
			noise_x = np.random.normal(0, R[0])
			noise_y = np.random.normal(0, R[1])
			noise_theta = np.random.normal(0, R[2])

			output[0] = output[0] + noise_x
			output[1] = output[1] + noise_y
			output[2] = output[2] + noise_theta

			next_state = output

			return next_state

		else:

			r_ICR = ((-2*l*vel[0]) / (vel[0]-vel[1])) + l

			omega = ((vel[0]-vel[1])*r) / (2*l) 


			matrix_A = np.array([ [cos(omega*dt) , -sin(omega*dt), 0],  
								  [sin(omega*dt) , cos(omega*dt) , 0],
								  [0             , 0             , 1]])


			x_tminus1 = np.array([ -r_ICR*sin(state[2]), 
								   r_ICR*cos(state[2]), 
								   state[2] ])

			matrix_B = np.array([state[0] + r_ICR*sin(state[2]),
								 state[1] - r_ICR*cos(state[2]),
								 omega * dt])

			immed = np.matmul(matrix_A, x_tminus1)
			output = immed + matrix_B

		
			#add noise
			noise_x = np.random.normal(0, R[0])
			noise_y = np.random.normal(0, R[1])
			noise_theta = np.random.normal(0, R[2])

			output[0] = output[0] + noise_x
			output[1] = output[1] + noise_y
			output[2] = output[2] + noise_theta

		
			next_state = output

	
		"*** YOUR CODE ENDS HERE ***"
		
		return next_state

	def dynamics_with_nonlinear_noise(self, state, vel, **kwargs):
		r"""
		Question 2(b)
		The dynamics of two-wheeled robot, be careful with the defined direction.
		
		Some parameters that you may use:
		@param dt: delta time
		@param r:  wheel radius
		@param l:  distancce from wheel to center
		@param vel:   2*1 matrix, the angular velocity of the right and left wheels, [omega1, omega2]
		@param state: 3*1 matrix, the state dimension, [x, y, theta (heading direction)]
		@param R:     2*1 matrix, noise amplitude of the additive Gaussian disturbances 
						for the angular velocity of wheels and heading, [R_omega, R_theta]. 
						Assume that the noises for omega1 and omega2 are the same.
		
		Return:
		@param next_state: 3*1 matrix, same as state
		"""
		dt = self.step_time
		r  = self.radius_wheel
		l  = self.radius
		R  = self.noise_amplitude
		
		"*** YOUR CODE STARTS HERE ***"

		next_state = np.zeros((1,3))

		#add noise
		noise_omega = np.random.normal(0, R[0])
		noise_theta = np.random.normal(0, R[1])

		vel[0] = vel[0] + noise_omega
		vel[1] = vel[1] + noise_omega
		state[2] = state[2] + noise_theta



		if vel[0] == vel[1]:
			#omega1 = omega2 --- the robot is walking on straight line, r_ICR is infinite, omega is 0

			output = np.array([state[0] + vel[0] * 1 * cos(state[2]) * dt, 
							   state[1] + vel[0] * 1 * sin(state[2]) * dt,
							   state[2] ])

			new_state = output

			return new_state

		else:

			r_ICR = ((-2*l*vel[0]) / (vel[0]-vel[1])) + l

			omega = ((vel[0]-vel[1])*r) / (2*l) 


			matrix_A = np.array([ [cos(omega*dt) , -sin(omega*dt), 0],  
								  [sin(omega*dt) , cos(omega*dt) , 0],
								  [0             , 0             , 1]])


			x_tminus1 = np.array([ -r_ICR*sin(state[2]), 
								   r_ICR*cos(state[2]), 
								   state[2] ])

			matrix_B = np.array([state[0] + r_ICR*sin(state[2]),
								 state[1] - r_ICR*cos(state[2]),
								 omega * dt])

			immed = np.matmul(matrix_A, x_tminus1)
			output = immed + matrix_B
		

			next_state = output

		"*** YOUR CODE ENDS HERE ***"
		
		return next_state


	def policy(self):
		r"""
		Question 3
		A simple policy for steering.

		Some parameters that you may use:
		@param dt: delta time
		@param r:  wheel radius
		@param l:  distancce from wheel to center

		Return:
		@instructions: A list containing instructions of wheels' angular velocities. 
					   Form: [[omega1_t1, omega2_t1], [omega1_t2, omega2_t2], ...]
		@timepoints: A list containing the duration time of each instruction.
					   Form: [t1, t2, ...], then the simulation time is \sum(t1+t2+...).
		@path_length: The shortest trajectory length after calculation by your hand.
		"""

		dt = self.step_time
		r  = self.radius_wheel
		l  = self.radius
		instructions = []
		timepoints = []
		path_length = 0
		
		"*** YOUR CODE STARTS HERE ***"
		


		"*** YOUR CODE ENDS HERE ***"
		
		return instructions, timepoints, path_length

	
	def plot_robot(self, ax, robot_color = 'g', goal_color='r', show_goal=True, show_text=False, show_traj=False, show_uncertainty=False, traj_type='-g', fontsize=10, **kwargs):
		x = self.state[0, 0]
		y = self.state[1, 0]

		goal_x = self.goal[0, 0]
		goal_y = self.goal[1, 0]

		robot_circle = mpl.patches.Circle(xy=(x, y), radius = self.radius, color = robot_color)
		robot_circle.set_zorder(3)

		ax.add_patch(robot_circle)
		if show_text: ax.text(x - 0.5, y, 'r'+ str(self.id), fontsize = fontsize, color = 'r')
		self.plot_patch_list.append(robot_circle)

		# arrow
		theta = self.state[2][0]
		arrow = mpl.patches.Arrow(x, y, 0.5*cos(theta), 0.5*sin(theta), width = 0.6)
		arrow.set_zorder(3)
		ax.add_patch(arrow)
		self.plot_patch_list.append(arrow)

		if show_goal:
			goal_circle = mpl.patches.Circle(xy=(goal_x, goal_y), radius = self.radius, color=goal_color, alpha=0.5)
			goal_circle.set_zorder(1)

			ax.add_patch(goal_circle)
			if show_text: ax.text(goal_x + 0.3, goal_y, 'g'+ str(self.id), fontsize = fontsize, color = 'k')
			self.plot_patch_list.append(goal_circle)

		if show_traj:
			x_list = [t[0, 0] for t in self.trajectory]
			y_list = [t[1, 0] for t in self.trajectory]
			self.plot_line_list.append(ax.plot(x_list, y_list, traj_type))


	def keyboard_to_angular(self, vel):
		r"""

		Change the velocity from [v, omega] to [omega1, omega2].
		
		Some parameters that you may use:
		@param r:  wheel radius
		@param l:  distancce from wheel to center
		@param vel: 2*1 matrix, the forward velocity and the rotation velocity [v, omega]

		Return:
		@param vel_new: 2*1 matrix,the angular velocity of right and left wheel, [omega1, omega2]
		"""
		l  = self.radius
		r  = self.radius_wheel
		
		vel_new = np.c_[[vel[0, 0]+vel[1, 0]*l, vel[0, 0]-vel[1, 0]*l]] / r
		return vel_new
	

