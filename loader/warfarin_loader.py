import pandas as pd
from data_loader import DataLoader

class WarfarinLoader(DataLoader):
	def __init__(self, file_path = "../data/warfarin.csv"):
		self.file_path = file_path

	def load_raw_data(self):
		self.df = pd.read_csv(file_path)

