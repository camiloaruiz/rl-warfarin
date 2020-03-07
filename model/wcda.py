from model.model import Model
import pandas as pd
import numpy as np
from loader.warfarin_loader import bin_weekly_dose_val

class WCDA(Model):
	def __init__(self, bin_weekly_dose):
		super().__init__(bin_weekly_dose)
		self.feature_columns = ["Age in decades", "Height in cm", "Weight in kg", "Asian race", "Black or African American", "Missing or Mixed race", "Enzyme inducer status", "Amiodarone status"]


	def featurize(self, wf):
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

	def predict(self, x, y):
		# Weekly dose
		coef = np.array([-0.2546, 0.0118, 0.0134, -0.6752, 0.4060, 0.0443, 1.2799, -0.5695])
		bias = 4.0376
		weekly_dose = (np.sum(coef*x) + bias)**2.00

		if not(self.bin_weekly_dose):
			out = weekly_dose
		else:
			out = bin_weekly_dose_val(weekly_dose)
		return out

	def get_true_Beta(self):
		raise NotImplementedError


	def expected_regrit(self, a_star_a_hat):
		if self.true_beta == None:
			self.true_beta = self.get_true_Beta()
 
		regret = []
		for a_star, a_hat in astar_ahat:
			raise NotImplementedError

		return regret