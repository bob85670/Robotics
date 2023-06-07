import numpy as np
import matplotlib as mpl
from math import sin, cos, atan2, inf, pi
from ir_sim.world import RobotBase
from ir_sim.global_param import env_param
from ir_sim.util.util import get_transform, WrapToPi

class RobotEKF(RobotBase):

	robot_type = 'custom'
	appearance = 'circle'
	state_dim  = (3, 1) # the state dimension, x, y, theta(heading direction),
	vel_dim = (2, 1)    # the angular velocity of right and left wheel
	goal_dim = (3, 1)	# the goal dimension, x, y, theta 
	position_dim = (2, 1) # the position dimension, x, y

	def __init__(self, id, state, vel = np.zeros((2, 1)), goal=np.zeros((3, 1)), 
				 step_time = 0.01, **kwargs):
		r""" FOR SETTING STARTS """
		self.shape  = kwargs.get('shape', [4.6, 1.6, 3, 1.6]) # Only for rectangle shape
		self.radius = kwargs.get('radius', 0.25)
		super(RobotEKF, self).__init__(id, state, vel, goal, step_time, **kwargs)
		r""" FOR SETTING ENDS """

		r""" FOR SIMULATION STARTS """
		self.landmark_map = self.get_landmark_map()
		# self.control_mode = kwargs.get('control_mode', 'auto') # 'auto' or 'policy'. Control the robot by keyboard or defined policy.

		self.s_mode  = kwargs.get('s_mode', 'sim') # 'sim', 'pre'. Plot simulate position or predicted position
		# self.s_mode   = kwargs.get('s_mode', 'none') # 'none', 'linear', 'nonlinear'. Simulation motion model with different noise mode
		self.s_R = kwargs.get('s_R', np.c_[[0.02, 0.02, 0.01]]) # Noise amplitude of simulation motion model
		r""" FOR SIMULATION ENDS """

		r""" FOR EKF ESTIMATION STARTS """
		self.e_state = {'mean': self.state, 'std': np.diag([1, 1, 1])}

		self.e_trajectory = []
		self.e_mode  = kwargs.get('e_mode', 'no_measure') # 'no_measure', 'no_bearing', 'bearing'. Estimation mode
		self.e_R     = kwargs.get('e_R', np.diag([0.02, 0.02, 0.01])) # Noise amplitude of ekf estimation motion model
		self.e_Q     = kwargs.get('e_Q', 0.2) # Noise amplitude of ekf estimation measurement model
		r""" FOR EKF ESTIMATION ENDS """

	def dynamics(self, state, vel, **kwargs):
		r"""
		Question 1
		The dynamics of two-wheeled robot for SIMULATION.

		Some parameters that you may use:
		@param dt:	  delta time
		@param vel  : 2*1 matrix, the forward velocity and the rotation velocity [v, omega]
		@param state: 3*1 matrix, the state dimension, [x, y, theta]
		@param noise: 3*1 matrix, noises of the additive Gaussian disturbances 
						for the state, [epsilon_x, epsilon_y, epsilon_theta]

		Return:
		@param next_state: 3*1 matrix, same as state
		"""
		dt     = self.step_time
		R_hat  = self.s_R
		noise  = np.random.normal(0, R_hat)

		"*** YOUR CODE STARTS HERE ***"

		#next_state = np.zeros((1,3))
		#print(next_state)

		next_state = np.array([state[0] + vel[0] * cos(state[2]) * dt, 
			                   state[1] + vel[0] * sin(state[2]) * dt,
			                   state[2] + vel[1] * dt])

		next_state = next_state + noise


		"*** YOUR CODE ENDS HERE ***"
		return next_state

	
	def ekf_predict(self, vel, **kwargs):
		r"""
		Question 2
		Predict the state of the robot.

		Some parameters that you may use:
		@param dt: delta time
		@param vel   : 2*1 matrix, the forward velocity and the rotation velocity [v, omega]
		@param mu    : 3*1 matrix, the mean of the belief distribution, [x, y, theta]
		@param sigma : 3*3 matrix, the covariance matrix of belief distribution.
		@param R     : 3*3 matrix, the assumed noise amplitude for dynamics, usually diagnal.

		Goal:
		@param mu_bar    : 3*1 matrix, the mean at the next time, as in EKF algorithm
		@param sigma_bar : 3*3 matrix, the covariance matrix at the next time, as in EKF algorithm
		"""
		dt = self.step_time
		R  = self.e_R
		mu = self.e_state['mean']
		sigma = self.e_state['std']
		
		"*** YOUR CODE STARTS HERE ***"
		# 1)Compute the Jacobian of G with respect to the state
		
		#print(vel[1,0].shape)
		#print(vel[1].shape)

		g_t = np.array([[1, 0, -vel[0,0] * sin(mu[2,0]) * dt],
						[0, 1, vel[0,0] * cos(mu[2,0]) * dt],
						[0, 0, 1]])

		#print(g_t.shape)


		# 2)Compute the mean 

		mu_bar = np.array([mu[0] + vel[0] * cos(mu[2]) * dt,
						   mu[1] + vel[0] * sin(mu[2]) * dt,
						   mu[2] + vel[1] * dt])


		# 3)Compute the covariance matrix

		immed1 = np.matmul(g_t, sigma)
		immed2 = np.matmul(immed1, g_t.transpose())
		sigma_bar = immed2 + R
		

		"*** YOUR CODE ENDS HERE ***"
		self.e_state['mean'] = mu_bar
		self.e_state['std'] = sigma_bar

	def ekf_correct_no_bearing(self, **kwargs):
		r"""
		Question 3
		Update the state of the robot using range measurement.
		
		Some parameters that you may use:
		@param dt: delta time
		@param mu_bar    : 3*1 matrix, the mean of the belief distribution, [x, y, theta]
		@param sigma_bar : 3*3 matrix, the covariance matrix of belief distribution.
		@param Q         : 1*1 matrix, the assumed noise amplitude for measurement, usually diagnal.
		@param lm_map    : a dict containing positions [x, y] of all the landmarks.
							Access by lm_map[landmark_id]
		@param lm_measurements : a list containing all the landmarks detected by sensor. 
							Each element is a dict, keys are 'id', 'range', 'angle'.
							Access by lm_measurements[0]['id'] (example).

		Goal:
		@param mu    : 3*1 matrix, the updated mean, as in EKF algorithm
		@param sigma : 3*3 matrix, the updated covariance matrix, as in EKF algorithm
		"""
		dt = self.step_time
		mu_bar = self.e_state['mean']
		sigma_bar = self.e_state['std']

		lm_map   = self.landmark_map
		lm_measurements = self.get_landmarks()
		Q = np.array([[self.e_Q]])

		#print(lm_map)
		#print(lm_measurements)

		for lm in lm_measurements:
			# Update mu_bar and sigma_bar with each measurement individually,
			"*** YOUR CODE STARTS HERE ***"
			
			# 1)Calculate the expected measurement vector

			#print(lm_map[lm['id']])		#position of landmarks
			#print(lm_map[lm['id']].shape)
			#print(lm_map[lm['id']][0,0])

			q = (lm_map[lm['id']][0,0] - mu_bar[0,0]) **2 + (lm_map[lm['id']][1,0] - mu_bar[1,0]) **2

			#print(q)

			# 2)Compute H
			
			a = np.array([ [-(lm_map[lm['id']][0,0] - mu_bar[0,0]), -(lm_map[lm['id']][1,0] - mu_bar[1,0]), 0] ])	#second row of H_t
			b = np.sqrt([1/q, 1/q, 1])

			H_t = np.multiply(a, b)

			#print(H_t.shape)
			#print(H_t)


			# 3)Gain of Kalman

			d = np.matmul(sigma_bar, H_t.transpose())
			e = np.matmul(H_t, d)
			f = e + Q
			g = np.linalg.inv(f)


			K_t = np.matmul(d, g)
			#print(K_t.shape)

			

			# 4)Kalman correction for mean_bar and covariance_bar

			#print(lm['range'])
			
			mu_bar = mu_bar + np.matmul(K_t, np.array([lm['range'] - np.sqrt([q])]))

			#print(sigma_bar)
			sigma_bar = np.matmul(np.identity(3) - np.matmul(K_t, H_t) , sigma_bar)
			#print(sigma_bar)
			#print(sigma_bar.shape)
			

			"*** YOUR CODE ENDS HERE ***"
			pass
		mu    = mu_bar
		sigma = sigma_bar
		self.e_state['mean'] = mu
		self.e_state['std'] = sigma

	def ekf_correct_with_bearing(self, **kwargs):
		r"""
		Question 4
		Update the state of the robot using range and bearing measurement.
		
		Some parameters that you may use:
		@param dt: delta time
		@param mu_bar    : 3*1 matrix, the mean of the belief distribution, [x, y, theta]
		@param sigma_bar : 3*3 matrix, the covariance matrix of belief distribution.
		@param Q         : 2*2 matrix, the assumed noise amplitude for measurement, usually diagnal.
		@param lm_map    : a dict containing positions [x, y] of all the landmarks.
							Access by lm_map[landmark_id]
		@param lm_measurements : a list containing all the landmarks detected by sensor. 
							Each element is a dict, keys are 'id', 'range', 'angle'.
							Access by lm_measurements[0]['id'] (example).
		
		Goal:
		@param mu    : 3*1 matrix, the updated mean, as in EKF algorithm
		@param sigma : 3*3 matrix, the updated covariance matrix, as in EKF algorithm
		"""
		dt = self.step_time
		mu_bar = self.e_state['mean']
		sigma_bar = self.e_state['std']

		lm_map    = self.landmark_map
		lm_measurements = self.get_landmarks()
		Q = np.diag([self.e_Q, self.e_Q])
		
		for lm in lm_measurements:
			# Update mu_bar and sigma_bar with each measurement individually,
			"*** YOUR CODE STARTS HERE ***"

			# Calculate the expected measurement vector
			q = (lm_map[lm['id']][0,0] - mu_bar[0,0]) **2 + (lm_map[lm['id']][1,0] - mu_bar[1,0]) **2


			# Compute H
			temp1 = np.array([ [(lm_map[lm['id']][1,0] - mu_bar[1,0])/q , -(lm_map[lm['id']][0,0] - mu_bar[0,0])/q , -1] ])

			a = np.array([ [-(lm_map[lm['id']][0,0] - mu_bar[0,0]), -(lm_map[lm['id']][1,0] - mu_bar[1,0]), 0] ])	#second row of H_t
			b = np.sqrt([1/q, 1/q, 1])
			c = np.multiply(a, b)

			H_t = np.vstack([c, temp1])


			# Gain of Kalman
			d = np.matmul(sigma_bar, H_t.transpose())
			e = np.matmul(H_t, d)
			f = e + Q
			g = np.linalg.inv(f)

			K_t = np.matmul(d, g)			


			# Kalman correction for mean_bar and covariance_bar
			h_mu_tbar = np.array([np.sqrt([q]),
								 [WrapToPi(np.arctan2(  (lm_map[lm['id']][1,0]-mu_bar[1,0]) , (lm_map[lm['id']][0,0]-mu_bar[0,0])  ) - mu_bar[2,0])]])

			z_t = np.array([[lm['range']], [lm['angle']]])

			mu_bar = mu_bar + np.matmul(K_t, z_t-h_mu_tbar)


			sigma_bar = np.matmul(np.identity(3) - np.matmul(K_t, H_t) , sigma_bar)



			"*** YOUR CODE ENDS HERE ***"
			pass
		mu = mu_bar
		sigma = sigma_bar
		self.e_state['mean'] = mu
		self.e_state['std'] = sigma
	
	
	def get_landmark_map(self, ):
		env_map = env_param.obstacle_list.copy()
		landmark_map = dict()
		for obstacle in env_map:
			if obstacle.landmark:
				landmark_map[obstacle.id] = obstacle.center[0:2]
		return landmark_map

	def post_process(self):
		self.ekf(self.vel)

	def ekf(self, vel):
		if self.s_mode == 'pre':
			if self.e_mode == 'no_measure':
				self.ekf_predict(vel)
				self.e_trajectory.append(self.e_state['mean'])
			elif self.e_mode == 'no_bearing':
				self.ekf_predict(vel)
				self.ekf_correct_no_bearing()
				self.e_trajectory.append(self.e_state['mean'])
			elif self.e_mode == 'bearing':
				self.ekf_predict(vel)
				self.ekf_correct_with_bearing()
				self.e_trajectory.append(self.e_state['mean'])
			else:
				raise ValueError('Not supported e_mode. Try \'no_measure\', \'no_bearing\', \'bearing\' for estimation mode.')
		elif self.s_mode == 'sim':
			pass
		else:
			raise ValueError('Not supported s_mode. Try \'sim\', \'pre\' for simulation mode.')

	def plot_robot(self, ax, robot_color = 'g', goal_color='r', 
					show_goal=True, show_text=False, show_uncertainty=False, 
					show_traj=False, traj_type='-g', fontsize=10, **kwargs):
		x = self.state[0, 0]
		y = self.state[1, 0]
		theta = self.state[2, 0]

		robot_circle = mpl.patches.Circle(xy=(x, y), radius = self.radius, color = robot_color, alpha = 0.5)
		robot_circle.set_zorder(3)
		ax.add_patch(robot_circle)
		if show_text: ax.text(x - 0.5, y, 'r'+ str(self.id), fontsize = fontsize, color = 'r')
		self.plot_patch_list.append(robot_circle)

		# arrow
		arrow = mpl.patches.Arrow(x, y, 0.5*cos(theta), 0.5*sin(theta), width = 0.6)
		arrow.set_zorder(3)
		ax.add_patch(arrow)
		self.plot_patch_list.append(arrow)

		if self.s_mode == 'pre':
			x = self.e_state['mean'][0, 0]
			y = self.e_state['mean'][1, 0]
			theta = self.e_state['mean'][2, 0]

			e_robot_circle = mpl.patches.Circle(xy=(x, y), radius = self.radius, color = 'y', alpha = 0.7)
			e_robot_circle.set_zorder(3)
			ax.add_patch(e_robot_circle)
			self.plot_patch_list.append(e_robot_circle)

			# calculate and plot covariance ellipse
			covariance = self.e_state['std'][:2, :2]
			eigenvals, eigenvecs = np.linalg.eig(covariance)

			# get largest eigenvalue and eigenvector
			max_ind = np.argmax(eigenvals)
			max_eigvec = eigenvecs[:,max_ind]
			max_eigval = eigenvals[max_ind]

			# get smallest eigenvalue and eigenvector
			min_ind = 0
			if max_ind == 0:
			    min_ind = 1

			min_eigvec = eigenvecs[:,min_ind]
			min_eigval = eigenvals[min_ind]

			# chi-square value for sigma confidence interval
			chisquare_scale = 2.2789  

			scale = 2
			# calculate width and height of confidence ellipse
			width = 2 * np.sqrt(chisquare_scale*max_eigval) * scale
			height = 2 * np.sqrt(chisquare_scale*min_eigval) * scale
			angle = np.arctan2(max_eigvec[1],max_eigvec[0])

			# generate covariance ellipse
			ellipse = mpl.patches.Ellipse(xy=[x, y], 
				width=width, height=height, 
				angle=angle/np.pi*180, alpha = 0.25)

			ellipse.set_zorder(1)
			ax.add_patch(ellipse)
			self.plot_patch_list.append(ellipse)

		if show_goal:
			goal_x = self.goal[0, 0]
			goal_y = self.goal[1, 0]

			goal_circle = mpl.patches.Circle(xy=(goal_x, goal_y), radius = self.radius, color=goal_color, alpha=0.5)
			goal_circle.set_zorder(1)

			ax.add_patch(goal_circle)
			if show_text: ax.text(goal_x + 0.3, goal_y, 'g'+ str(self.id), fontsize = fontsize, color = 'k')
			self.plot_patch_list.append(goal_circle)

		if show_traj:
			x_list = [t[0, 0] for t in self.trajectory]
			y_list = [t[1, 0] for t in self.trajectory]
			self.plot_line_list.append(ax.plot(x_list, y_list, traj_type))
			
			if self.s_mode == 'pre':
				x_list = [t[0, 0] for t in self.e_trajectory]
				y_list = [t[1, 0] for t in self.e_trajectory]
				self.plot_line_list.append(ax.plot(x_list, y_list, '-y'))

