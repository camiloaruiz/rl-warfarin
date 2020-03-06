from model.model import Model
import pandas as pd
import numpy as np
from loader.warfarin_loader import bin_weekly_dose_val

class WPDA(Model):
	def __init__(self, bin_weekly_dose):
		super().__init__(bin_weekly_dose)
		self.feature_columns = ["Age in decades", "Height in cm", "Weight in kg", "VKORC1 A/G", "VKORC1 A/A", "VKORC1 genotype unknown", "CYP2C9 *1/*2", "CYP2C9 *1/*3", "CYP2C9*2/*2", "CYP2C9*2/*3", "CYP2C9*3/*3", "CYP2C9 genotype unknown", "Asian race", "Black or African American", "Missing or Mixed race", "Enzyme inducer status", "Amiodarone status"]

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

	def predict(self, x, y):
		# Weekly dose
		coef = np.array([-0.2614, 0.0087, 0.0128, -0.8677, -1.6974, -0.4854, -0.5211, -0.9357, -1.0616, -1.9206, -2.3312, -0.2188, -0.1092, -0.2760, -0.1032, 1.1816, -0.5503])
		bias = 5.6044
		weekly_dose = (np.sum(coef*x) + bias)**2.0

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