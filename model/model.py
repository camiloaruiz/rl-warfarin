import random
import numpy as np
import pandas as pd


class Model():
	def __init__(self, bin_weekly_dose, n_features=10):
		self.bin_weekly_dose = bin_weekly_dose
		if (self.bin_weekly_dose):
			self.out_column = "Binned weekly warfarin dose"
		else:
			self.out_column = "Weekly warfarin dose"

		# these are features sorted by importance from running feature_selection.py
		self.feature_columns = ['Weight in kg', 'VKORC1_1542_CC', 'VKORC1 A/A', 'Asian race', 'Height in cm', 'Black or African American', 'Smoker', 'Age in decades', 'CYP2C9 *1/*3', 'VKORC1_497_TT', 'White race', 'Enzyme inducer status', 'VKORC1 A/G', 'VKORC1_1542_CG', 'CYP2C9*2/*3', 'VKORC1_497_GG', 'Diabetes', 'VKORC1_1542_NA', 'Aspirin', 'VKORC1 genotype unknown', 'VKORC1_4451_CC', 'CYP2C9 *1/*2', 'VKORC1_4451_AC', 'Simvastatin', 'VKORC1_497_unknown', 'VKORC1_4451_AA', 'Amiodarone status', 'Congestive Heart Failure', 'CYP2C9 *1/*1', 'VKORC1_4451_NA', 'VKORC1_497_GT', 'Valve replacement', 'CYP2C9*3/*3', 'is Female', 'Missing or Mixed race', 'is Male', 'CYP2C9*2/*2', 'unknown Gender', 'CYP2C9 genotype unknown']
		self.feature_columns = self.feature_columns[:n_features]



	def get_X(self):
		return self.X


	def get_Y(self):
		return self.Y


	def set_X(self, X):
		self.X = X


	def set_Y(self, Y):
		self.Y = Y


	def remove_rows_with_missing_data(self):
		self.feat_df = self.feat_df.dropna(axis = 'rows')


	def prepare_XY(self):
		self.remove_rows_with_missing_data()
		self.set_X(self.feat_df[self.feature_columns].values)
		self.set_Y(self.feat_df[self.out_column].values)


	def predict(self, x, y):
		raise NotImplementedError


	def train(self, x, y):
		raise NotImplementedError


	def get_true_Beta(self):
		raise NotImplementedError


	def expected_regrit(self, a_star_a_hat):
		raise NotImplementedError

	def regret_over_time(self, a_star_a_hat):
		frac_incorrect = []
		a_star, a_hat = map(list, zip(*a_star_a_hat))
		regret = 1.0-np.equal(a_star, a_hat).astype(int)
		return np.cumsum(regret)

	def calc_frac_incorrect(self, a_star_a_hat):
		frac_incorrect = []
		a_star, a_hat = map(list, zip(*a_star_a_hat))
		for i in range(1, len(a_star_a_hat)+1):
			frac_incorrect.append(1.0-(np.sum(np.equal(a_star[:i], a_hat[:i]))/float(i)))
		return frac_incorrect

	def calc_final_frac_incorrect(self, a_star_a_hat):
		a_star, a_hat = map(list, zip(*a_star_a_hat))
		return 1.0 - np.sum(np.equal(a_star, a_hat))/float(len(a_star))



	# a_star_a_hat is list of Tuples 
	# [(a*_1, a^_1), ... ,(a*_i, a^_i), ... ,(a*_T, a^_T)]
	def experiment(self, rand_seed = 1):
		X, Y = self.X, self.Y
		assert(X.shape[0] == Y.shape[0])

		data = list(zip(X, Y))
		random.seed(rand_seed)
		random.shuffle(data)

		a_star_a_hat = []
		for x, y in data:

			if np.any(np.isin(x,np.nan)) or np.any(np.isin(y,np.nan)) or np.any(np.isin(x,'na')) or np.any(np.isin(y,'na')):

				x = np.where(x==np.nan, 0, x) 
				x = np.where(x=='na', 0, x) 
				
			if np.any(np.isin(y,np.nan)) or np.any(np.isin(y,'na')):
				continue

			a = self.predict(x,y)
			a_star_a_hat.append((y, a))
		return a_star_a_hat







	def featurize(self, wf):
		self.featurize_full(wf)
		columns = self.feature_columns + [self.out_column]
		self.feat_df = self.feat_df[columns]



	def featurize_full(self, wf):
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
		self.feat_df["VKORC1_1542_NA"] = wf.get_VKORC1_1542_NA()

		self.feat_df["VKORC1_4451_CC"] = wf.get_VKORC1_4451_CC()
		self.feat_df["VKORC1_4451_AC"] = wf.get_VKORC1_4451_AC()
		self.feat_df["VKORC1_4451_AA"] = wf.get_VKORC1_4451_AA()
		self.feat_df["VKORC1_4451_NA"] = wf.get_VKORC1_4451_NA()


		self.feat_df["VKORC1 A/G"] = wf.get_VKORC1_AG()
		self.feat_df["VKORC1 A/A"] = wf.get_VKORC1_AA()
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


