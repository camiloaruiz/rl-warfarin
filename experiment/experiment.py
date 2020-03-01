class Experiment():
	def __init__(self, model):
		self.model = model

	def run(self, evaluation):
		X, Y = self.model.X, self.model.Y
		assert(X.shape[0] == Y.shape[0])
		
		Y_hat = []
		for x, y, i in zip(X, Y, range(0, len(X))):
			y_hat = model.predict(x)
			Y_hat.append(y_hat)

			evaluation.run(Y_hat, Y[:i, :])


