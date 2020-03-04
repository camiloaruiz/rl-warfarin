
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

	def predict(self, x, **kw):
		raise NotImplementedError