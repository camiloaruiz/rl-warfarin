import numpy as np
from model.model import Model
import pandas as pd
from loader.warfarin_loader import bin_weekly_dose_val



class UCBDNet(Model):
	def __init__(self, bin_weekly_dose, num_actions=3, bound_constant=2.0, num_force=1.0):
		super().__init__(bin_weekly_dose)
		self.feature_columns = ["Age in decades", "Height in cm", "Weight in kg", "VKORC1 A/G", "VKORC1 A/A", "VKORC1 genotype unknown", "CYP2C9 *1/*2", "CYP2C9 *1/*3", "CYP2C9*2/*2", "CYP2C9*2/*3", "CYP2C9 genotype unknown", "Asian race", "Black or African American", "Missing or Mixed race", "Enzyme inducer status", "Amiodarone status"]

		self.dim = len(self.feature_columns) 
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


	def get_true_Beta(self):
		raise NotImplementedError


	def expected_regrit(self, a_star_a_hat):
		if self.true_beta == None:
			self.true_beta = self.get_true_Beta()
 
		regret = []
		for a_star, a_hat in astar_ahat:
			raise NotImplementedError

		return regret


	def featurize(self, wf):
		self.feat_df = pd.DataFrame()
		self.feat_df["Age in decades"] = wf.get_age_in_decades()
		self.feat_df["Height in cm"] = wf.get_height_in_cm()
		self.feat_df["Weight in kg"] = wf.get_weight_in_kg()
		self.feat_df["VKORC1 A/G"] = wf.get_VKORC1_AG()
		self.feat_df["VKORC1 A/A"] = wf.get_VKORC1_AA()
		self.feat_df["VKORC1 genotype unknown"] = wf.get_VKORC1_genotype_unknown()
		self.feat_df["CYP2C9 *1/*2"] = wf.get_CYP2C9_12()
		self.feat_df["CYP2C9 *1/*3"] = wf.get_CYP2C9_13()
		self.feat_df["CYP2C9*2/*2"] = wf.get_CYP2C9_22()
		self.feat_df["CYP2C9*2/*3"] = wf.get_CYP2C9_23()
		self.feat_df["CYP2C9*3/*3"] = wf.get_CYP2C9_33()
		self.feat_df["CYP2C9 genotype unknown"] = wf.get_CYP2C9_genotype_unknown()
		self.feat_df["Asian race"] = wf.get_asian_race()
		self.feat_df["Black or African American"] = wf.get_black_or_african_american()
		self.feat_df["Missing or Mixed race"] = wf.get_missing_or_mixed_race()
		self.feat_df["Enzyme inducer status"] = wf.get_enzyme_inducer_status()
		self.feat_df["Amiodarone status"] = wf.get_amiodarone_status()
		self.feat_df["Weekly warfarin dose"] = wf.get_weekly_warfarin_dose()
		if (self.bin_weekly_dose):
			self.feat_df[self.out_column] = wf.get_binned_weekly_warfarin_dose()


