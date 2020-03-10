import numpy as np
from model.model import Model
import pandas as pd
from loader.warfarin_loader import bin_weekly_dose_val



class ThompsonNet(Model):
	def __init__(self, bin_weekly_dose, num_actions=3, R=0.5, delta=0.1, epsilon=1.0/np.log(1000), num_force=0.0):
		super().__init__(bin_weekly_dose)
		#self.feature_columns = ["Age in decades", "Height in cm", "Weight in kg", "VKORC1 A/G", "VKORC1 A/A", "VKORC1 genotype unknown", "CYP2C9 *1/*2", "CYP2C9 *1/*3", "CYP2C9*2/*2", "CYP2C9*2/*3", "CYP2C9*3/*3", "CYP2C9 genotype unknown", "Asian race", "Black or African American", "Missing or Mixed race", "Enzyme inducer status", "Amiodarone status"]

		self.dim = len(self.feature_columns) +1
		self.num_actions = num_actions
		self.actions = np.identity(self.num_actions, dtype=float)
		self.true_beta = None
		self.R = R
		self.delta = delta
		self.epsilon = epsilon
		self.v2 = (self.R**2) * (24.0/self.epsilon) * self.dim *self.num_actions* np.log(1.0/self.delta)
		self.B = np.identity(self.dim*self.num_actions, dtype=float)
		self.mu = np.zeros((self.dim*self.num_actions,1))
		self.f = np.zeros((self.dim*self.num_actions,1))
		self.counts = np.zeros((self.num_actions))
		self.num_force = num_force


	def set_X(self, X):
		self.X = np.insert(X, 0, 1, axis=1)


	def predict(self, x, y):
		x.astype(float)
		y.astype(int)
		r_estimates = []
		theta = np.random.multivariate_normal(np.squeeze(self.mu), self.v2*np.linalg.inv(self.B))
		theta = np.expand_dims(theta, axis=1)
		for a in range(self.num_actions):
			if self.counts[a] < self.num_force:
				self.counts[a] += 1.0
				self.train(x, y, a)
				return a 

			x_a = np.outer(self.actions[a], x).flatten()
			x_a = np.expand_dims(x_a, axis=1).astype(float)
			r_estimates.append(np.matmul(theta.T, x_a))
		a = np.argmax(r_estimates)
		self.train(x, y, a)
		return a 


	def train(self, x, y, a):
		x_a = np.outer(self.actions[a], x).flatten()
		x_a = np.expand_dims(x_a, axis=1).astype(float)
		self.B += np.matmul(x_a, x_a.T) 
		self.f += self.reward(y, a)*x_a
		self.mu = np.matmul(np.linalg.inv(self.B), self.f)


	def reward(self, y, a):
		if y == a:
			return 0
		else:
			return -1


class ThompsonDNet(Model):
	def __init__(self, bin_weekly_dose, num_actions=3, R=0.5, delta=0.1, epsilon=1.0/np.log(1000), num_force=0.0):
		super().__init__(bin_weekly_dose)
		#self.feature_columns = ["Age in decades", "Height in cm", "Weight in kg", "VKORC1 A/G", "VKORC1 A/A", "VKORC1 genotype unknown", "CYP2C9 *1/*2", "CYP2C9 *1/*3", "CYP2C9*2/*2", "CYP2C9*2/*3", "CYP2C9*3/*3", "CYP2C9 genotype unknown", "Asian race", "Black or African American", "Missing or Mixed race", "Enzyme inducer status", "Amiodarone status"]

		self.dim = len(self.feature_columns) +1 
		self.num_actions = num_actions
		self.actions = np.identity(self.num_actions, dtype=float)
		self.true_beta = None
		self.R = R
		self.delta = delta
		self.epsilon = epsilon
		self.v2 = (self.R**2) * (24.0/self.epsilon) * self.dim * np.log(1.0/self.delta)
		print("v**2: ", self.v2)
		self.B = []
		self.mu = []
		self.f = []
		for i in range(self.num_actions):
			self.B.append(np.identity(self.dim, dtype=float))
			self.mu.append(np.zeros((self.dim,1)))
			self.f.append(np.zeros((self.dim,1)))
		self.counts = np.zeros((self.num_actions))
		self.num_force = num_force


	def set_X(self, X):
		self.X = np.insert(X, 0, 1, axis=1)


	def predict(self, x, y):
		x.astype(float)
		x = np.expand_dims(x, axis=1).astype(float)
		y.astype(int)
		r_estimates = []
		for a in range(self.num_actions):
			if self.counts[a] < self.num_force:
				self.counts[a] += 1.0
				self.train(x, y, a)
				return a 

			theta = np.random.multivariate_normal(np.squeeze(self.mu[a]), self.v2*np.linalg.inv(self.B[a]))
			theta = np.expand_dims(theta, axis=1)
			r_estimates.append(np.matmul(theta.T, x))
		a = np.argmax(r_estimates)
		self.train(x, y, a)
		return a 


	def train(self, x, y, a):
		x.astype(float)
		self.B[a] += np.matmul(x, x.T) 
		self.f[a] += self.reward(y, a)*x
		self.mu[a] = np.matmul(np.linalg.inv(self.B[a]), self.f[a])


	def reward(self, y, a):
		if y == a:
			return 0
		else:
			return -1

