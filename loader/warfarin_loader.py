import pandas as pd
import numpy as np

def get_enzyme_inducer_status_helper(row):
	carbamazepine = row["Carbamazepine (Tegretol)"]
	phenytoin = row["Phenytoin (Dilantin)"]
	rifampin = row["Rifampin or Rifampicin"]

	if (pd.isnull(carbamazepine) or pd.isnull(phenytoin) or pd.isnull(rifampin)):
		enzyme_inducer_status = np.nan
	else:
		enzyme_inducer_status = carbamazepine or phenytoin or rifampin

	return enzyme_inducer_status

class WarfarinLoader():
	def __init__(self, file_path = "data/warfarin.csv"):
		self.file_path = file_path
		self.load_raw_data()

	def load_raw_data(self):
		self.raw_df = pd.read_csv(self.file_path, )
		self.raw_df = self.raw_df.dropna(how = 'all') # Drop rows where all entries are nan
		self.raw_df = self.raw_df.fillna("na")

	def impute(self):
		raise NotImplementedError

	def get_age_in_decades(self):
		age2decade = {'10 - 19': 1, '20 - 29': 2, '30 - 39': 3, '40 - 49': 4, '50 - 59': 5, '60 - 69': 6, '70 - 79': 7, '80 - 89': 8, '90+': 9, 'na': np.nan}
		age_in_decades = self.raw_df["Age"].apply(lambda age: age2decade[age])
		return age_in_decades

	def get_height_in_cm(self):
		height_in_cm = self.raw_df["Height (cm)"]
		return height_in_cm

	def get_weight_in_kg(self):
		weight_in_kg = self.raw_df["Weight (kg)"]
		return weight_in_kg

	def binarize_feature(self, col, map_dict):
		assert(np.nansum(list(map_dict.values())) == 1) # Should be one hot
		return self.raw_df[col].apply(lambda genotype: map_dict[genotype])

	def get_VKORC1_AG(self):
		return self.binarize_feature("VKORC1 -1639 consensus", {"A/G": 1, "A/A": 0, "G/G": 0, "na": np.nan})

	def get_VKORC1_AA(self):
		return self.binarize_feature("VKORC1 -1639 consensus", {"A/G": 0, "A/A": 1, "G/G": 0, "na": np.nan})

	def get_VKORC1_genotype_unknown(self):
		return self.binarize_feature("VKORC1 -1639 consensus", {"A/G": 0, "A/A": 0, "G/G": 0, "na": 1})

	def get_CYP2C9_12(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':0, '*1/*3':0, '*1/*2':1, '*2/*2':0, '*2/*3':0, '*3/*3':0, "na":np.nan, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})

	def get_CYP2C9_13(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':0, '*1/*3':1, '*1/*2':0, '*2/*2':0, '*2/*3':0, '*3/*3':0, "na":np.nan, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})

	def get_CYP2C9_22(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':0, '*1/*3':0, '*1/*2':0, '*2/*2':1, '*2/*3':0, '*3/*3':0, "na":np.nan, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})

	def get_CYP2C9_23(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':0, '*1/*3':0, '*1/*2':0, '*2/*2':0, '*2/*3':1, '*3/*3':0, "na":np.nan, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})

	def get_CYP2C9_33(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':0, '*1/*3':0, '*1/*2':0, '*2/*2':0, '*2/*3':0, '*3/*3':1, "na":np.nan, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})

	def get_CYP2C9_genotype_unknown(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':0, '*1/*3':0, '*1/*2':0, '*2/*2':0, '*2/*3':0, '*3/*3':0, "na":1, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})

	def get_asian_race(self):
		return self.binarize_feature("Race", {'White':0, 'Unknown':0, 'Black or African American':0, 'Asian':1})

	def get_black_or_african_american(self):
		return self.binarize_feature("Race", {'White':0, 'Unknown':0, 'Black or African American':1, 'Asian':0})

	def get_missing_or_mixed_race(self):
		return self.binarize_feature("Race", {'White':0, 'Unknown':1, 'Black or African American':0, 'Asian':0})

	def get_enzyme_inducer_status(self):
		return self.raw_df.apply(lambda row: get_enzyme_inducer_status_helper(row), axis = 1)

	def get_amiodarone_status(self):
		return self.binarize_feature("Amiodarone (Cordarone)", {0.:0, 1.:1., "na":np.nan})

	def get_weekly_warfarin_dose(self):
		return self.wf["Therapeutic Dose of Warfarin"]