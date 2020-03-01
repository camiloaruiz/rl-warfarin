class Model():
	def __init__(self):
		raise NotImplementedError

	def get_X(self):
		return self.X

	def get_Y(self):
		return self.Y

	def set_X(self, X):
		self.X = X

	def set_Y(self, Y):
		self.Y = Y

	def prepare_XY(self, feat_df):
		raise NotImplementedError

	def predict(self, x):
		raise NotImplementedError