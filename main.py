if __name__ == "__main__":
	# Get data
	wf = WarfarinLoader()
	
	# Instantiate model and process data
	model = FixedDose()
	model.prepare_XY()	

	#
