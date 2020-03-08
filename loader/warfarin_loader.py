import pandas as pd
import numpy as np

def get_enzyme_inducer_status_helper(row,na_val=np.nan):
	carbamazepine = row["Carbamazepine (Tegretol)"]
	phenytoin = row["Phenytoin (Dilantin)"]
	rifampin = row["Rifampin or Rifampicin"]

	enzyme_inducer_status = 0
	if na_val==np.nan:
		if ((carbamazepine == "na") or (phenytoin == "na") or (rifampin == "na")):
			enzyme_inducer_status = 0
		else:
			enzyme_inducer_status = carbamazepine or phenytoin or rifampin

	else:

		if ((carbamazepine == 1) or (phenytoin == 1) or (rifampin == 1)):
			enzyme_inducer_status = 1
		else:
			enzyme_inducer_status = 0

	return enzyme_inducer_status

def bin_weekly_dose_val(weekly_dose_val):
	if weekly_dose_val == "na":
		out = np.nan
	elif weekly_dose_val < 21:
		out = 0
	elif 21 <= weekly_dose_val <= 49:
		out = 1
	elif weekly_dose_val > 49:
		out = 2
	else:
		assert(False)
	return out

class WarfarinLoader():
	def __init__(self, file_path = "data/warfarin.csv",na_val=np.nan,fill_na_mean=False,stable_dose_only=False):
		self.file_path = file_path
		self.load_raw_data()
		self.na_val = na_val  #this value is populated into the binary genotype and drug choices to allow reverting back to initial
		self.fill_na_mean = fill_na_mean
		self.stable_dose_only = stable_dose_only

	def load_raw_data(self):
		self.raw_df = pd.read_csv(self.file_path, )
		self.raw_df = self.raw_df.dropna(how = 'all') # Drop rows where all entries are nan
		self.raw_df = self.raw_df.fillna("na")

	def impute(self):
		raise NotImplementedError


	def binarize_feature(self, col, map_dict):
		assert(np.nansum(list(map_dict.values())) == 1) # Should be one hot
		return self.raw_df[col].apply(lambda genotype: map_dict[genotype])




	def get_is_male(self):
		return self.binarize_feature("Gender", {'male': 1, 'female': 0,'na':self.na_val})

	def get_is_female(self):
		return self.binarize_feature("Gender", {'male': 0, 'female': 1,'na':self.na_val})

	def get_gender_unknown(self):
		return self.binarize_feature("Gender", {'male': 0, 'female': 0,'na':1.})

	def get_age_in_decades(self):
		age2decade = {'10 - 19': 1, '20 - 29': 2, '30 - 39': 3, '40 - 49': 4, '50 - 59': 5, '60 - 69': 6, '70 - 79': 7, '80 - 89': 8, '90+': 9, 'na': np.nan}
		age_in_decades = self.raw_df["Age"].apply(lambda age: age2decade[age])
		if not self.fill_na_mean: return age_in_decades
		mean = np.nanmean(age_in_decades)
		#print("mean age",mean)
		return age_in_decades.replace(np.nan,mean)

	def get_height_in_cm(self):
		height_in_cm = self.raw_df["Height (cm)"].replace("na",np.nan)
		if not self.fill_na_mean: return height_in_cm
		mean = np.nanmean(height_in_cm)
		#print("mean height",mean)
		return height_in_cm.replace(np.nan,mean)

	def get_weight_in_kg(self):
		weight_in_kg = self.raw_df["Weight (kg)"].replace("na",np.nan)
		if not self.fill_na_mean: return weight_in_kg
		mean = np.nanmean(weight_in_kg)
		#print("mean weight",mean)
		return weight_in_kg.replace(np.nan,mean)




	def get_diabetes(self):
		return self.binarize_feature("Diabetes", {"na": self.na_val, 1.: 1, 0.: 0})

	def get_valve_replacement(self):
		return self.binarize_feature("Valve Replacement", {"na": self.na_val, 1.: 1, 0.: 0})

	def get_CHF(self):
		return self.binarize_feature("Congestive Heart Failure and/or Cardiomyopathy", {"na": self.na_val, 1.: 1, 0.: 0})

	def get_aspirin(self):
		return self.binarize_feature("Aspirin", {"na": self.na_val, 1.: 1, 0.: 0})

	def get_simvastatin(self):
		return self.binarize_feature("Simvastatin (Zocor)", {"na": self.na_val, 1.: 1, 0.: 0})

	def get_smoker(self):
		return self.binarize_feature("Current Smoker", {"na": self.na_val, 1.: 1, 0.: 0})



	def get_VKORC1_497_GT(self):
		return self.binarize_feature("VKORC1 497 consensus", {"G/T": 1, "T/T": 0, "G/G":0 ,"na": self.na_val})
	def get_VKORC1_497_TT(self):
		return self.binarize_feature("VKORC1 497 consensus", {"G/T": 0, "T/T": 1, "G/G":0 ,"na": self.na_val})
	def get_VKORC1_497_GG(self):
		return self.binarize_feature("VKORC1 497 consensus", {"G/T": 0, "T/T": 0, "G/G":1 ,"na": self.na_val})

	def get_VKORC1_497_unknown(self):
		return self.binarize_feature("VKORC1 497 consensus", {"G/T": 0, "T/T": 0, "G/G":0 ,"na": 1})


	def get_VKORC1_1542_CC(self):
		return self.binarize_feature("VKORC1 1542 consensus", {"C/C": 1, "C/G": 0, "G/G": 0, "na": self.na_val})
	def get_VKORC1_1542_CG(self):
		return self.binarize_feature("VKORC1 1542 consensus", {"C/C": 0, "C/G": 1, "G/G": 0, "na": self.na_val})
	def get_VKORC1_1542_GG(self):
		return self.binarize_feature("VKORC1 1542 consensus", {"C/C": 0, "C/G": 0, "G/G": 1, "na": self.na_val})
	def get_VKORC1_1542_NA(self):
		return self.binarize_feature("VKORC1 1542 consensus", {"C/C": 0, "C/G": 0, "G/G": 0, "na": 1})


	def get_VKORC1_4451_CC(self):
		return self.binarize_feature("VKORC1 -4451 consensus", {"C/C": 1, "A/C": 0, "A/A": 0,"na": self.na_val})
	def get_VKORC1_4451_AC(self):
		return self.binarize_feature("VKORC1 -4451 consensus", {"C/C": 0, "A/C": 1, "A/A": 0,"na": self.na_val})
	def get_VKORC1_4451_AA(self):
		return self.binarize_feature("VKORC1 -4451 consensus", {"C/C": 0, "A/C": 0, "A/A": 1, "na": self.na_val})
	def get_VKORC1_4451_NA(self):
		return self.binarize_feature("VKORC1 -4451 consensus", {"C/C": 0, "A/C": 0, "A/A": 0,"na": 1})





	def get_VKORC1_AG(self):
		return self.binarize_feature("VKORC1 -1639 consensus", {"A/G": 1, "A/A": 0, "G/G": 0, "na": self.na_val})

	def get_VKORC1_AA(self):
		return self.binarize_feature("VKORC1 -1639 consensus", {"A/G": 0, "A/A": 1, "G/G": 0, "na": self.na_val})

	def get_VKORC1_genotype_unknown(self):
		return self.binarize_feature("VKORC1 -1639 consensus", {"A/G": 0, "A/A": 0, "G/G": 0, "na": 1})

	def get_CYP2C9_11(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':1, '*1/*3':0, '*1/*2':0, '*2/*2':0, '*2/*3':0, '*3/*3':0, "na":self.na_val, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})

	def get_CYP2C9_12(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':0, '*1/*3':0, '*1/*2':1, '*2/*2':0, '*2/*3':0, '*3/*3':0, "na":self.na_val, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})

	def get_CYP2C9_13(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':0, '*1/*3':1, '*1/*2':0, '*2/*2':0, '*2/*3':0, '*3/*3':0, "na":self.na_val, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})

	def get_CYP2C9_22(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':0, '*1/*3':0, '*1/*2':0, '*2/*2':1, '*2/*3':0, '*3/*3':0, "na":self.na_val, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})

	def get_CYP2C9_23(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':0, '*1/*3':0, '*1/*2':0, '*2/*2':0, '*2/*3':1, '*3/*3':0, "na":self.na_val, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})

	def get_CYP2C9_33(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':0, '*1/*3':0, '*1/*2':0, '*2/*2':0, '*2/*3':0, '*3/*3':1, "na":self.na_val, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})

	def get_CYP2C9_genotype_unknown(self):
		return self.binarize_feature("CYP2C9 consensus", {'*1/*1':0, '*1/*3':0, '*1/*2':0, '*2/*2':0, '*2/*3':0, '*3/*3':0, "na":1, '*1/*5':0, '*1/*13':0, '*1/*14':0, '*1/*11':0, '*1/*6':0})



	def get_asian_race(self):
		return self.binarize_feature("Race", {'White':0, 'Unknown':0, 'Black or African American':0, 'Asian':1})

	def get_black_or_african_american(self):
		return self.binarize_feature("Race", {'White':0, 'Unknown':0, 'Black or African American':1, 'Asian':0})

	def get_missing_or_mixed_race(self):
		return self.binarize_feature("Race", {'White':0, 'Unknown':1, 'Black or African American':0, 'Asian':0})

	def get_white_race(self):
		return self.binarize_feature("Race", {'White':1, 'Unknown':0, 'Black or African American':0, 'Asian':0})







	def get_enzyme_inducer_status(self):
		return self.raw_df.apply(lambda row: get_enzyme_inducer_status_helper(row,self.na_val), axis = 1)

	def get_amiodarone_status(self):
		return self.binarize_feature("Amiodarone (Cordarone)", {0.:0, 1.:1., "na":self.na_val})

	def get_is_stable_dose(self):
		return self.binarize_feature("Subject Reached Stable Dose of Warfarin",{0.:0, 1.:1., "na":0.})

	#careful that raw data is being modified to "na" if stable dose is not reached
	def get_weekly_warfarin_dose(self):
		weekly = self.raw_df["Therapeutic Dose of Warfarin"]

		if self.stable_dose_only:
			stable = self.get_is_stable_dose()
			for i in range(len(weekly)):
				if stable[i] == 0:
					weekly[i] = np.nan

		return weekly

	def get_binned_weekly_warfarin_dose(self):
		weekly = self.get_weekly_warfarin_dose().replace(np.nan,"na")
		return weekly.apply(lambda weekly_dose_val: bin_weekly_dose_val(weekly_dose_val))















