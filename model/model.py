import random
import numpy as np
import pandas as pd


class Model():
	def __init__(self, bin_weekly_dose, n_features=20):
		self.bin_weekly_dose = bin_weekly_dose
		if (self.bin_weekly_dose):
			self.out_column = "Binned weekly warfarin dose"
		else:
			self.out_column = "Weekly warfarin dose"
		self.true_beta = None
		self.num_actions = 3

		# these are features sorted by importance from running feature_selection.py
		#self.feature_columns = ['Weight in kg', 'VKORC1_1542_CC', 'VKORC1 A/A', 'Asian race', 'Height in cm', 'Black or African American', 'Smoker', 'Age in decades', 'CYP2C9 *1/*3', 'VKORC1_497_TT', 'White race', 'Enzyme inducer status', 'VKORC1 A/G', 'VKORC1_1542_CG', 'CYP2C9*2/*3', 'VKORC1_497_GG', 'Diabetes', 'VKORC1_1542_NA', 'Aspirin', 'VKORC1 genotype unknown', 'VKORC1_4451_CC', 'CYP2C9 *1/*2', 'VKORC1_4451_AC', 'Simvastatin', 'VKORC1_497_unknown', 'VKORC1_4451_AA', 'Amiodarone status', 'Congestive Heart Failure', 'CYP2C9 *1/*1', 'VKORC1_4451_NA', 'VKORC1_497_GT', 'Valve replacement', 'CYP2C9*3/*3', 'is Female', 'Missing or Mixed race', 'is Male', 'CYP2C9*2/*2', 'unknown Gender', 'CYP2C9 genotype unknown']
		
		
		#correlation with rewards instead
		#self.feature_columns = ['Weight in kg', 'VKORC1 A/G', 'Enzyme inducer status', 'Black or African American', 'VKORC1 G/G', 'VKORC1_497_GT', 'Age in decades', 'CYP2C9 *1/*2', 'Missing or Mixed race', 'VKORC1_1542_CG', 'VKORC1_1542_GG', 'Amiodarone status', 'White race', 'Congestive Heart Failure', 'CYP2C9 *1/*3', 'CYP2C9*2/*3', 'VKORC1_497_GG', 'VKORC1_497_TT', 'Smoker', 'Diabetes', 'VKORC1 A/A', 'Asian race', 'Aspirin', 'CYP2C9 *1/*1', 'Valve replacement', 'CYP2C9*3/*3', 'VKORC1_4451_CC', 'VKORC1_4451_AC', 'Height in cm', 'VKORC1_4451_AA', 'VKORC1_1542_CC', 'CYP2C9*2/*2', 'is Female', 'Simvastatin', 'is Male']				
		
		#self.feature_columns = self.feature_columns[:n_features]
		
		
		#original columns
		self.feature_columns = ["Age in decades", "Height in cm", "Weight in kg", "VKORC1 A/G", "VKORC1 A/A", "VKORC1 genotype unknown", "CYP2C9 *1/*2", "CYP2C9 *1/*3", "CYP2C9*2/*2", "CYP2C9*2/*3", "CYP2C9*3/*3", "CYP2C9 genotype unknown", "Asian race", "Black or African American", "Missing or Mixed race", "Enzyme inducer status", "Amiodarone status"]



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
		# self.remove_rows_with_missing_data()
		self.set_X(self.feat_df[self.feature_columns].values)
		self.set_Y(self.feat_df[self.out_column].values)


	def predict(self, x, y):
		raise NotImplementedError


	def train(self, x, y):
		raise NotImplementedError


	# y is the binned y value of 0,1,2,etc
	def get_y_for_action(self,a,y):
		return (a==y).astype(float) - 1


	#returns list of true betas
	def get_true_Beta(self):
		# print(self.X.shape)
		# X = self.X[~np.isnan(self.X).any(axis=1)]
		# print(X.shape)

		X = self.X
		Y = self.Y
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


	def expected_regret(self, a_star_a_hat):
		if self.true_beta == None:
			self.true_beta = self.get_true_Beta()
		betas = self.true_beta
		regret_step = []
		for i, (a_star, a_hat) in enumerate(a_star_a_hat):
			rs = [np.dot(self.X[i],betas[j]) for j in range(self.num_actions)]
			r = max(rs) - rs[int(a_hat)]
			# r = np.dot(self.X[i],betas[int(a_star)]) - np.dot(self.X[i],betas[int(a_hat)])
			regret_step.append(r)
		return np.cumsum(regret_step)


	def calc_frac_incorrect(self, a_star_a_hat):
		frac_incorrect = []
		a_star, a_hat = map(list, zip(*a_star_a_hat))
		for i in range(1, len(a_star_a_hat)+1):
			frac_incorrect.append(1.0-(np.sum(np.equal(a_star[:i], a_hat[:i]))/float(i)))
		return frac_incorrect



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

			if np.any(np.isin(x,np.nan)) or np.any(np.isin(x,'na')):
				continue
				x = np.where(x==np.nan, 0, x) 
				x = np.where(x=='na', 0, x) 
				
			if np.any(np.isin(y,np.nan)) or np.any(np.isin(y,'na')):
				continue

			a = self.predict(x,y)
			a_star_a_hat.append((y, a))
		return a_star_a_hat







	def featurize(self, wf, remove_nas_before_selecting_columns = True):
		self.featurize_full(wf)
		columns = self.feature_columns + [self.out_column]
		# if remove_nas_before_selecting_columns:
		# 	self.remove_rows_with_missing_data() 
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


