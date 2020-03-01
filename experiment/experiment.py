# TODO: Make the main function an experiment class obj
# Then implement logging to easy to track results across runs (and pickle everything together to keep)

# class Experiment():
# 	def __init__(self, model):
# 		self.model = model

# 	def run(self, evaluation):
# 		X, Y = self.model.X, self.model.Y
# 		assert(X.shape[0] == Y.shape[0])
		
# 		Y_hat = []
# 		for x, y, i in zip(X, Y, range(0, len(X))):
# 			y_hat = model.predict(x)
# 			Y_hat.append(y_hat)

# 			evaluation.run(Y_hat, Y[:i, :])


