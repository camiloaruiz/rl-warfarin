import numpy as np
from model.model import Model
import pandas as pd
from loader.warfarin_loader import bin_weekly_dose_val



class testModel( Model):
	def __init__(self, bin_weekly_dose, e_0=0.1, num_actions=3, num_force=1.0):
		super().__init__(bin_weekly_dose)
		self.feature_columns = ["is Male","is Female","unknown Gender","Age in decades", "Height in cm", "Weight in kg",
		"Diabetes","Valve replacement","Congestive Heart Failure","Aspirin","Simvastatin","Smoker",
		"VKORC1_497_GT","VKORC1_497_TT","VKORC1_497_GG","VKORC1_497_unknown",
		"VKORC1_1542_CC","VKORC1_1542_CG","VKORC1_1542_GG","VKORC1_1542_NA",
		"VKORC1_4451_CC","VKORC1_4451_AC","VKORC1_4451_AA","VKORC1_4451_NA",
		"VKORC1 A/G", "VKORC1 A/A", "VKORC1 G/G","VKORC1 genotype unknown",
		"CYP2C9 *1/*1","CYP2C9 *1/*2", "CYP2C9 *1/*3", "CYP2C9*2/*2", "CYP2C9*2/*3","CYP2C9*3/*3", "CYP2C9 genotype unknown",
		"Asian race", "Black or African American", "Missing or Mixed race","White race",
		"Enzyme inducer status", "Amiodarone status"]

		self.dim = len(self.feature_columns) 
		self.num_actions = num_actions
		#self.bound_constant = bound_constant
		self.actions = np.identity(self.num_actions, dtype=float)
		self.true_beta = None
		self.A = np.identity(self.dim*self.num_actions, dtype=float)
		self.b = np.zeros((self.dim*self.num_actions,1))
		self.counts = np.zeros((self.num_actions))
		self.num_force = num_force
		self.e = e_0    #epsilon in the epsilon greedy! must be
		
	def featurize(self, wf):
		self.feat_df = pd.DataFrame()
		self.feat_df["is Male"] = wf.get_is_male()
		self.feat_df["is Female"] = wf.get_is_female()
		self.feat_df["unknown Gender"] = wf.get_gender_unknown()
		self.feat_df["Age in decades"] = wf.get_age_in_decades()
		self.feat_df["Height in cm"] = wf.get_height_in_cm()
		self.feat_df["Weight in kg"] = wf.get_weight_in_kg()

		self.feat_df["Diabetes"] = wf.get_diabetes()
		self.feat_df["Valve replacement"] = wf.get_valve_replacement()
		self.feat_df["Congestive Heart Failure"] = wf.get_CHF()
		self.feat_df["Aspirin"] = wf.get_aspirin()
		self.feat_df["Simvastatin"] = wf.get_simvastatin()
		self.feat_df["Smoker"] = wf.get_smoker()

		self.feat_df["VKORC1_497_GT"] = wf.get_VKORC1_497_GT()
		self.feat_df["VKORC1_497_TT"] = wf.get_VKORC1_497_TT()
		self.feat_df["VKORC1_497_GG"] = wf.get_VKORC1_497_GG()
		self.feat_df["VKORC1_497_unknown"] = wf.get_VKORC1_497_unknown()

		self.feat_df["VKORC1_1542_CC"] = wf.get_VKORC1_1542_CC()
		self.feat_df["VKORC1_1542_CG"] = wf.get_VKORC1_1542_CG()
		self.feat_df["VKORC1_1542_GG"] = wf.get_VKORC1_1542_GG()
		self.feat_df["VKORC1_1542_NA"] = wf.get_VKORC1_1542_NA()

		self.feat_df["VKORC1_4451_CC"] = wf.get_VKORC1_4451_CC()
		self.feat_df["VKORC1_4451_AC"] = wf.get_VKORC1_4451_AC()
		self.feat_df["VKORC1_4451_AA"] = wf.get_VKORC1_4451_AA()
		self.feat_df["VKORC1_4451_NA"] = wf.get_VKORC1_4451_NA()


		self.feat_df["VKORC1 A/G"] = wf.get_VKORC1_AG()
		self.feat_df["VKORC1 A/A"] = wf.get_VKORC1_AA()
		self.feat_df["VKORC1 G/G"] = wf.get_VKORC1_GG()
		self.feat_df["VKORC1 genotype unknown"] = wf.get_VKORC1_genotype_unknown()

		self.feat_df["CYP2C9 *1/*1"] = wf.get_CYP2C9_11()
		self.feat_df["CYP2C9 *1/*2"] = wf.get_CYP2C9_12()
		self.feat_df["CYP2C9 *1/*3"] = wf.get_CYP2C9_13()
		self.feat_df["CYP2C9*2/*2"] = wf.get_CYP2C9_22()
		self.feat_df["CYP2C9*2/*3"] = wf.get_CYP2C9_23()
		self.feat_df["CYP2C9*3/*3"] = wf.get_CYP2C9_33()
		self.feat_df["CYP2C9 genotype unknown"] = wf.get_CYP2C9_genotype_unknown()

		self.feat_df["Asian race"] = wf.get_asian_race()
		self.feat_df["Black or African American"] = wf.get_black_or_african_american()
		self.feat_df["Missing or Mixed race"] = wf.get_missing_or_mixed_race()
		self.feat_df["White race"] = wf.get_white_race()

		self.feat_df["Enzyme inducer status"] = wf.get_enzyme_inducer_status()
		self.feat_df["Amiodarone status"] = wf.get_amiodarone_status()
		self.feat_df["Weekly warfarin dose"] = wf.get_weekly_warfarin_dose()
		if (self.bin_weekly_dose):
			self.feat_df[self.out_column] = wf.get_binned_weekly_warfarin_dose()


"""
	def predict(self, x, y):
		x.astype(float)
		y.astype(int)

		if np.random.uniform() < self.e: #random case
			a = np.random.choice(range(self.num_actions))
			self.counts[a] += 1.0
			self.train(x, y, a)
			return a

		#greedy case
		theta = np.matmul(np.linalg.inv(self.A), self.b)
		r_estimates = []
		for a in range(self.num_actions):
			if self.counts[a] < self.num_force:
				self.counts[a] += 1.0
				self.train(x, y, a)
				return a 

			x_a = np.outer(self.actions[a], x).flatten()
			x_a = np.expand_dims(x_a, axis=1).astype(float)
			r_estimates.append(np.matmul(theta.T, x_a)) #+ self.bound_constant*np.sqrt(np.matmul(np.matmul(x_a.T,np.linalg.inv(self.A)), x_a)))
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


	def get_true_Beta(self):
		raise NotImplementedError


	def expected_regrit(self, a_star_a_hat):
		if self.true_beta == None:
			self.true_beta = self.get_true_Beta()
 
		regret = []
		for a_star, a_hat in astar_ahat:
			raise NotImplementedError

		return regret

	def featurize_2(self, wf):
		self.feat_df = pd.DataFrame()
		self.feat_df["Age in decades"] = wf.get_age_in_decades()
		self.feat_df["Height in cm"] = wf.get_height_in_cm()
		self.feat_df["Weight in kg"] = wf.get_weight_in_kg()
		self.feat_df["Asian race"] = wf.get_asian_race()
		self.feat_df["Black or African American"] = wf.get_black_or_african_american()
		self.feat_df["Missing or Mixed race"] = wf.get_missing_or_mixed_race()
		self.feat_df["Enzyme inducer status"] = wf.get_enzyme_inducer_status()
		self.feat_df["Amiodarone status"] = wf.get_amiodarone_status()
		self.feat_df["Weekly warfarin dose"] = wf.get_weekly_warfarin_dose()
		if (self.bin_weekly_dose):
			self.feat_df[self.out_column] = wf.get_binned_weekly_warfarin_dose()
"""



