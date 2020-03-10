import numpy as np
from model.model import Model
import pandas as pd
from loader.warfarin_loader import bin_weekly_dose_val



class UCBNet( Model):
	def __init__(self, bin_weekly_dose, num_actions=3, bound_constant=2.0, num_force=1.0):
		super().__init__(bin_weekly_dose)
		#self.feature_columns = ["Age in decades", "Height in cm", "Weight in kg", "VKORC1 A/G", "VKORC1 A/A", "VKORC1 genotype unknown", "CYP2C9 *1/*2", "CYP2C9 *1/*3", "CYP2C9*2/*2", "CYP2C9*2/*3", "CYP2C9*3/*3", "CYP2C9 genotype unknown", "Asian race", "Black or African American", "Missing or Mixed race", "Enzyme inducer status", "Amiodarone status"]

		self.dim = len(self.feature_columns) + 1
		self.num_actions = num_actions
		self.bound_constant = bound_constant
		self.actions = np.identity(self.num_actions, dtype=float)
		self.true_beta = None
		self.A = np.identity(self.dim*self.num_actions, dtype=float)
		self.b = np.zeros((self.dim*self.num_actions,1))
		self.counts = np.zeros((self.num_actions))
		self.num_force = num_force


	def set_X(self, X):
		nan_ = np.isnan(X).astype(float)
		for i in range(len(self.feature_columns)):
			print(self.feature_columns[i], np.sum(nan_[:,i]))


		X_mean = np.nanmean(X, axis=0)
		for i in range(len(self.feature_columns)):
			X[:,i] = np.where(np.isnan(X[:,i]), X_mean[i], X[:,i]) 
			# X[:,i] = np.where(np.isnan(X[:,i]), 0.0, X[:,i]) 

		self.X = np.insert(X, 0, 1, axis=1)
		

	def predict(self, x, y):
		x.astype(float)
		y.astype(int)
		theta = np.matmul(np.linalg.inv(self.A), self.b) 
		r_estimates = []
		for a in range(self.num_actions):
			if self.counts[a] < self.num_force:
				self.counts[a] += 1.0
				self.train(x, y, a)
				return a 

			x_a = np.outer(self.actions[a], x).flatten()
			x_a = np.expand_dims(x_a, axis=1).astype(float)
			r_estimates.append(np.matmul(theta.T, x_a)+ self.bound_constant*np.sqrt(np.matmul(np.matmul(x_a.T,np.linalg.inv(self.A)), x_a)))
		a = np.argmax(r_estimates)
		self.train(x, y, a)
		return a 
			

	def train(self, x, y, a):
		x_a = np.outer(self.actions[a], x).flatten()
		x_a = np.expand_dims(x_a, axis=1).astype(float)
		self.A += np.matmul(x_a, x_a.T)
		self.b += self.reward(y, a)*x_a


	def reward(self, y, a):
		if y == a:
			return 0.0
		else:
			return -1.0



class UCBDNet(Model):
	def __init__(self, bin_weekly_dose, num_actions=3, bound_constant=2.0, num_force=1.0):
		super().__init__(bin_weekly_dose)
		#self.feature_columns = ["Age in decades", "Height in cm", "Weight in kg", "VKORC1 A/G", "VKORC1 A/A", "VKORC1 genotype unknown", "CYP2C9 *1/*2", "CYP2C9 *1/*3", "CYP2C9*2/*2", "CYP2C9*2/*3", "CYP2C9*3/*3", "CYP2C9 genotype unknown", "Asian race", "Black or African American", "Missing or Mixed race", "Enzyme inducer status", "Amiodarone status"]

		self.dim = len(self.feature_columns) +1
		self.num_actions = num_actions
		self.bound_constant = bound_constant
		self.true_beta = None
		self.A = []
		self.b = []
		self.counts = np.zeros((self.num_actions))
		self.num_force = num_force
		for i in range(self.num_actions):
			self.A.append(np.identity(self.dim, dtype=float))
			self.b.append(np.zeros((self.dim,1)))


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

			theta = np.matmul(np.linalg.inv(self.A[a]), self.b[a]) 
			r_estimates.append(np.matmul(theta.T, x)+ self.bound_constant*np.sqrt(np.matmul(np.matmul(x.T,np.linalg.inv(self.A[a])), x)))
		a = np.argmax(r_estimates)
		self.train(x, y, a)
		return a 


	def train(self, x, y, a):
		self.A[a] += np.matmul(x, x.T) 
		self.b[a] += self.reward(y, a)*x


	def reward(self, y, a):
		if y == a:
			return 0.0
		else:
			return -1.0
