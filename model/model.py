import random
import numpy as np
import pandas as pd


class Model():
	def __init__(self, bin_weekly_dose=3, feature_group=0, impute_VKORC1=True):
		self.bin_weekly_dose = bin_weekly_dose
		self.impute_VKORC1 = impute_VKORC1
		if (self.bin_weekly_dose>=2):
			self.out_column = "Binned weekly warfarin dose"
		else:
			self.out_column = "Weekly warfarin dose"
		self.true_beta = None
		self.num_actions = bin_weekly_dose

		# these are features sorted by importance from running feature_selection.py
		#self.feature_columns = ['Weight in kg', 'VKORC1_1542_CC', 'VKORC1 A/A', 'Asian race', 'Height in cm', 'Black or African American', 'Smoker', 'Age in decades', 'CYP2C9 *1/*3', 'VKORC1_497_TT', 'White race', 'Enzyme inducer status', 'VKORC1 A/G', 'VKORC1_1542_CG', 'CYP2C9*2/*3', 'VKORC1_497_GG', 'Diabetes', 'VKORC1_1542_NA', 'Aspirin', 'VKORC1 genotype unknown', 'VKORC1_4451_CC', 'CYP2C9 *1/*2', 'VKORC1_4451_AC', 'Simvastatin', 'VKORC1_497_unknown', 'VKORC1_4451_AA', 'Amiodarone status', 'Congestive Heart Failure', 'CYP2C9 *1/*1', 'VKORC1_4451_NA', 'VKORC1_497_GT', 'Valve replacement', 'CYP2C9*3/*3', 'is Female', 'Missing or Mixed race', 'is Male', 'CYP2C9*2/*2', 'unknown Gender', 'CYP2C9 genotype unknown']
		
		if feature_group==0:
			# wpda
			self.feature_columns = ["Age in decades", "Height in cm", "Weight in kg", "VKORC1 A/G", "VKORC1 A/A", "VKORC1 genotype unknown", "CYP2C9 *1/*2", "CYP2C9 *1/*3", "CYP2C9*2/*2", "CYP2C9*2/*3", "CYP2C9*3/*3", "CYP2C9 genotype unknown", "Asian race", "Black or African American", "Missing or Mixed race", "Enzyme inducer status", "Amiodarone status"]
		else:
			# almost all of them 
			self.feature_columns = ['Weight in kg', 'VKORC1 A/G', 'Enzyme inducer status', 'Black or African American', 'VKORC1 G/G', 'VKORC1_497_GT', 'Age in decades', 'CYP2C9 *1/*2', 'Missing or Mixed race', 'VKORC1_1542_CG', 'VKORC1_1542_GG', 'Amiodarone status', 'White race', 'Congestive Heart Failure', 'CYP2C9 *1/*3', 'CYP2C9*2/*3', 'VKORC1_497_GG', 'VKORC1_497_TT', 'Smoker', 'Diabetes', 'VKORC1 A/A', 'Asian race', 'Aspirin', 'CYP2C9 *1/*1', 'Valve replacement', 'CYP2C9*3/*3', 'VKORC1_4451_CC', 'VKORC1_4451_AC', 'Height in cm', 'VKORC1_4451_AA', 'VKORC1_1542_CC', 'CYP2C9*2/*2', 'is Female', 'Simvastatin', 'is Male']				

	def get_X(self):
		return self.X

	def get_Y(self):
		return self.Y

	def set_X(self, X):
		X_mean = np.nanmean(X, axis=0)
		for i in range(len(self.feature_columns)):
			# X[:,i] = np.where(np.isnan(X[:,i]), X_mean[i], X[:,i]) 
			X[:,i] = np.where(np.isnan(X[:,i]), 0.0, X[:,i]) 

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

	# For training 
	def reward(self, y, a, y_hat):
		if self.num_actions == 1:
			return y - float(y_hat)
		else:
			if y == a:
				return 0.0
			else:
				return -1.0

	# For training 
	def return_binner(self, a, y_hat):
		if self.num_actions == 1:
			# print(type(y_hat.item()))
			return float(y_hat)
		else:
			return a

	# y is the binned y value of 0,1,2,etc
	def get_y_for_action(self,a,y):
		return (a==y).astype(float) - 1

	#returns list of true betas
	def calc_true_Beta(self):
		# print(self.get_X().shape)
		# X = self.get_X()[~np.isnan(self.get_X()).any(axis=1)]
		# print(X.shape)

		X = self.get_X()
		Y = self.get_Y()
		betas = []
		for a in range(self.num_actions):
			y_a = self.get_y_for_action(a,Y)
			beta = np.linalg.lstsq(X,y_a)[0]
			betas.append(beta)

		#testing to see if they work
		"""
		incorrect = []
		num_incorrect = 0
		count = 0
		for i in range(len(Y)):
			x,y = X[i],Y[i]
			a = np.argmax([np.dot(x, betas[j]) for j in range(self.num_actions)])
			#print(a,[np.dot(x, betas[j]) for j in range(self.num_actions)])
			if a != y:
				print(a,y,[np.dot(x, betas[j]) for j in range(self.num_actions)])
				incorrect.append(0 - np.dot(x, betas[int(a)]))
				num_incorrect += 1
			count +=1
		print(np.mean(incorrect))
		print("True beta score: ", num_incorrect/count)
		"""
		#exit()
		return np.array(betas)

	def set_true_Beta(self):
		self.true_beta = self.calc_true_Beta()

	def get_true_Beta(self):
		return self.true_beta

	def expected_regret_one_step(self, x, a_hat):
		betas = self.get_true_Beta()
		true_reward_estimates = [np.dot(x,betas[j]) for j in range(self.num_actions)]
		regret_one_step = max(true_reward_estimates) - true_reward_estimates[int(a_hat)]
		return regret_one_step

	def observed_regret_one_step(self, a_star, a_hat):
		if a_star == a_hat:
			observed_regret = 0.0
		else: 
			observed_regret = 1.0
		return observed_regret

	def non_binned_regret(self, a_star_a_hat):
		regret_step = []
		for i, (a_star, a_hat) in enumerate(a_star_a_hat):
			r = np.abs(a_star-a_hat)
			regret_step.append(r)
		return np.cumsum(regret_step)

	def calc_frac_correct(self, a_star_a_hat):
		frac_incorrect = []
		a_star, a_hat = map(list, zip(*a_star_a_hat))
		for i in range(1, len(a_star_a_hat)+1):
			frac_incorrect.append((np.mean(np.equal(a_star[:i], a_hat[:i])) ))
		return frac_incorrect

	def calc_frac_incorrect(self, a_star_a_hat):
		frac_incorrect = []
		a_star, a_hat = map(list, zip(*a_star_a_hat))
		for i in range(1, len(a_star_a_hat)+1):
			frac_incorrect.append((1.0 - np.mean(np.equal(a_star[:i], a_hat[:i])) ))
		return frac_incorrect

	def non_binned_calc_frac_incorrect(self, a_star_a_hat):
		frac_incorrect = []
		a_star, a_hat = map(list, zip(*a_star_a_hat))
		for i in range(1, len(a_star_a_hat)+1):
			frac_incorrect.append(np.mean(np.abs(np.array(a_star[:i])-np.array(a_hat[:i]))))
			# frac_incorrect.append(np.mean(np.abs(np.array(a_star[:i])-np.array(a_hat[:i]))/np.array(a_star[:i])))
		return frac_incorrect

	# a_star_a_hat is list of Tuples 
	# [(a*_1, a^_1), ... ,(a*_i, a^_i), ... ,(a*_T, a^_T)]
	def experiment(self, rand_seed = 1):
		# Get Data
		X, Y = self.get_X(), self.get_Y()

		# Get true betas
		self.set_true_Beta()
		
		# Shuffle data
		assert(len(X) == len(Y))
		np.random.seed(rand_seed+67958254)
		random_order = np.random.permutation(len(X))
		X = X[random_order]
		Y = Y[random_order]
		
		# Run model
		a_star_a_hat = []
		expected_regret_step_list = []
		observed_regret_step_list = []

		# y == a_star
		for x, y in zip(X, Y): #y is a_star; y_hat is the estimate of reward for the action chosen
			a = self.predict(x,y)
			a_star_a_hat.append((y, a))
			expected_regret_step_list.append(self.expected_regret_one_step(x, a))
			observed_regret_step_list.append(self.observed_regret_one_step(y, a))
		
		cum_expected_regret = np.cumsum(expected_regret_step_list)
		cum_observed_regret = np.cumsum(observed_regret_step_list)

		return a_star_a_hat, cum_expected_regret, cum_observed_regret

	def featurize(self, wf, remove_nas_before_selecting_columns = True):
		self.featurize_full(wf)
		columns = self.feature_columns + [self.out_column]
		self.feat_df = self.feat_df[columns]

	def featurize_full(self, wf):
		self.feat_df = pd.DataFrame()
		if (self.impute_VKORC1):
			wf.impute_VKORC1()

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
		if (self.bin_weekly_dose>=2):
			self.feat_df[self.out_column] = wf.get_binned_weekly_warfarin_dose(self.bin_weekly_dose)