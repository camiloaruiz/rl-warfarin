from model.model import Model
import pandas as pd

from loader.warfarin_loader import bin_weekly_dose_val, bin_weekly_dose_val_2, bin_weekly_dose_val_4, bin_weekly_dose_val_5


class FixedDose(Model):
	def __init__(self, bin_weekly_dose, feature_group=0):
		super().__init__(bin_weekly_dose)
		#self.feature_columns = []

#	def featurize(self, wf):
#		self.feat_df = pd.DataFrame()
#		self.feat_df["Weekly warfarin dose"] = wf.get_weekly_warfarin_dose()
#		if (self.bin_weekly_dose):
#			self.feat_df[self.out_column] = wf.get_binned_weekly_warfarin_dose()

	def predict(self, x, y):
		weekly_dose = 35
		if self.bin_weekly_dose ==2:
			out = bin_weekly_dose_val_2(weekly_dose)
		elif self.bin_weekly_dose == 3:
			out = bin_weekly_dose_val(weekly_dose)
		elif self.bin_weekly_dose == 4:
			out = bin_weekly_dose_val_4(weekly_dose)
		elif self.bin_weekly_dose == 5:
			out = bin_weekly_dose_val_5(weekly_dose)
		else:
			out = weekly_dose
		return out


