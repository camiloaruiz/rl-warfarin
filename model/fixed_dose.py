from model.model import Model
import pandas as pd

class FixedDose(Model):
	def __init__(self, bin_weekly_dose):
		super().__init__(bin_weekly_dose)
		self.feature_columns = []

	def featurize(self, wf):
		self.feat_df = pd.DataFrame()
		self.feat_df["Weekly warfarin dose"] = wf.get_weekly_warfarin_dose()
		if (self.bin_weekly_dose):
			self.feat_df[self.out_column] = wf.get_binned_weekly_warfarin_dose()

	def predict(self, x):
		assert(self.bin_weekly_dose)
		return 1