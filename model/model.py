import random
import numpy as np


class Model():
	def __init__(self, bin_weekly_dose):
		self.bin_weekly_dose = bin_weekly_dose
		if (self.bin_weekly_dose):
			self.out_column = "Binned weekly warfarin dose"
		else:
			self.out_column = "Weekly warfarin dose"


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


	def calc_frac_incorrect(self, a_star_a_hat):
		frac_incorrect = []
		a_star, a_hat = map(list, zip(*a_star_a_hat))
		for i in range(1, len(a_star_a_hat)+1):
			frac_incorrect.append(1.0-(np.sum(np.equal(a_star[:i], a_hat[:i]))/float(len(a_star_a_hat))))
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

			if np.any(np.isin(x,np.nan)) or np.any(np.isin(y,np.nan)) or np.any(np.isin(x,'na')) or np.any(np.isin(y,'na')):
				continue

			a = self.predict(x,y)
			a_star_a_hat.append((y, a))
		return a_star_a_hat


