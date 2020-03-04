import numpy as np

class Evaluation():
	def __init__(self, evaluation_metrics):
		self.evaluation_metrics = evaluation_metrics
		self.frac_incorrect = []
		self.regret = []

	def get_frac_incorrect(self):
		return self.frac_incorrect

	def get_regret(self):
		return self.regret

	def calc_frac_incorrect(self, Y_hat, Y):
		assert(len(Y_hat) == len(Y))
		frac_incorrect = 1-np.sum(np.equal(Y_hat, Y))/float(len(Y))
		self.frac_incorrect.append(frac_incorrect)

	def calc_regret(self):
		assert(len(Y_hat) == len(Y))
		regret = np.sum(np.not_equal(Y_hat, Y))
		self.regret.append(regret)

	def evaluate(self, Y_hat, Y):
		assert(Y_hat.shape == Y.shape)
		if "frac_incorrect" in self.evaluation_metrics:
			self.calc_frac_incorrect(Y_hat, Y)
		if "regret" in self.evaluation_metrics:
			self.calc_regret()