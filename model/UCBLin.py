import numpy as np
from model.model import Model
import pandas as pd
from loader.warfarin_loader import bin_weekly_dose_val

class UCBNet( Model):
	def __init__(self, bin_weekly_dose, num_actions=3, bound_constant=2.0, num_force=1.0, feature_group=0, impute_VKORC1 = True):
		super().__init__(bin_weekly_dose = bin_weekly_dose, feature_group = feature_group, impute_VKORC1 = impute_VKORC1)
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

	def predict(self, x, y):
		x = np.append(x, 1.0) 
		x.astype(float)
		# y.astype(int)
		theta = np.matmul(np.linalg.inv(self.A), self.b) 
		r_estimates = []
		for a in range(self.num_actions):
			if self.counts[a] < self.num_force:
				self.counts[a] += 1.0
				y_hat = 0.0
				self.train(x, y, a, 0.0)
				return self.return_binner(a, y_hat)

			x_a = np.outer(self.actions[a], x).flatten()
			x_a = np.expand_dims(x_a, axis=1).astype(float)
			r_estimates.append(np.matmul(theta.T, x_a)+ self.bound_constant*np.sqrt(np.matmul(np.matmul(x_a.T,np.linalg.inv(self.A)), x_a)))
		a = np.argmax(r_estimates)
		y_hat = r_estimates[a]
		self.train(x, y, a, y_hat)
		return self.return_binner(a, y_hat)

	def train(self, x, y, a, y_hat):
		x_a = np.outer(self.actions[a], x).flatten()
		x_a = np.expand_dims(x_a, axis=1).astype(float)
		self.A += np.matmul(x_a, x_a.T)
		self.b += self.reward(y, a, y_hat)*x_a

class UCBDNet(Model):
	def __init__(self, bin_weekly_dose, num_actions=3, bound_constant=2.0, num_force=1.0, feature_group=0, impute_VKORC1 = True):
		super().__init__(bin_weekly_dose = bin_weekly_dose, feature_group = feature_group, impute_VKORC1 = impute_VKORC1)
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

	def predict(self, x, y):
		x = np.append(x, 1.0) 
		x.astype(float)
		x = np.expand_dims(x, axis=1).astype(float)
		# y.astype(int)
		r_estimates = []
		for a in range(self.num_actions):
			if self.counts[a] < self.num_force:
				self.counts[a] += 1.0
				y_hat = 0.0
				self.train(x, y, a, 0.0)
				return self.return_binner(a, y_hat)

			theta = np.matmul(np.linalg.inv(self.A[a]), self.b[a]) 
			r_estimates.append(np.matmul(theta.T, x)+ self.bound_constant*np.sqrt(np.matmul(np.matmul(x.T,np.linalg.inv(self.A[a])), x)))
		a = np.argmax(r_estimates)
		y_hat = r_estimates[a]
		self.train(x, y, a, y_hat)
		return self.return_binner(a, y_hat)

	def train(self, x, y, a, y_hat):
		self.A[a] += np.matmul(x, x.T) 
		self.b[a] += self.reward(y, a, y_hat)*x