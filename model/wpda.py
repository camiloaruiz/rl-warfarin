from model.model import Model
import pandas as pd

class WPDA(Model):
	def __init__(self):
		pass

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