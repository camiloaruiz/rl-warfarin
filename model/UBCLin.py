import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Module
from model.model import Model
import pandas as pd
from loader.warfarin_loader import bin_weekly_dose_val



class UCBNet(Module, Model):
    def __init__(self, bin_weekly_dose, num_actions=3, bound_constant=2):
    	Module.__init__(self)
    	Model.__init__(self, bin_weekly_dose)
        self.feature_columns = ["Age in decades", "Height in cm", "Weight in kg", "VKORC1 A/G", "VKORC1 A/A", "VKORC1 genotype unknown", "CYP2C9 *1/*2", "CYP2C9 *1/*3", "CYP2C9*2/*2", "CYP2C9*2/*3", "CYP2C9 genotype unknown", "Asian race", "Black or African American", "Missing or Mixed race", "Enzyme inducer status", "Amiodarone status"]

        self.dim = len(self.feature_columns) 

        self.num_actions = num_actions
        self.counts = torch.zeros([self.num_actions], dtype=torch.float32,requires_grad=False)
		self.register_buffer('a_counts', self.counts)

		self.bound_constant = torch.tensor([bound_constant], dtype=torch.float32,requires_grad=False)
		self.register_buffer('bound_constant', self.bound_constant)

        self.lin = -1*F.sigmoid(nn.Linear(in_features=self.dim, out_features=self.num_actions, bias=True))

    def forward(self, x, t):
    	x = self.lin(x)
    	for i in range(self.num_actions):
    		if self.counts[i] == 0.0: 
    			self.counts[i] += 1.0
    			return x[i], i
    	time = torch.tensor([t], dtype=torch.float32,requires_grad=False)
    	UCbounds = torch.sqrt(self.bound_constant*troch.div(torch.log(time)*torch.ones([self.num_actions], dtype=torch.float32,requires_grad=False),self.counts))
    	i = torch.argmax(x + UCbounds).item()
    	self.counts[i] += 1.0
        return torch.max(x + UCbounds), i 

    def predict(self, x, t):
    	 self.forward(x, t)

    def featurize(self, wf):
        self.feat_df = pd.DataFrame()
        self.feat_df["Age in decades"] = wf.get_age_in_decades()
        self.feat_df["Height in cm"] = wf.get_height_in_cm()
        self.feat_df["Weight in kg"] = wf.get_weight_in_kg()
        self.feat_df["VKORC1 A/G"] = wf.get_VKORC1_AG()
        self.feat_df["VKORC1 A/A"] = wf.get_VKORC1_AA()
        self.feat_df["VKORC1 genotype unknown"] = wf.get_VKORC1_genotype_unknown()
        self.feat_df["CYP2C9 *1/*2"] = wf.get_CYP2C9_12()
        self.feat_df["CYP2C9 *1/*3"] = wf.get_CYP2C9_13()
        self.feat_df["CYP2C9*2/*2"] = wf.get_CYP2C9_22()
        self.feat_df["CYP2C9*2/*3"] = wf.get_CYP2C9_23()
        self.feat_df["CYP2C9*3/*3"] = wf.get_CYP2C9_33()
        self.feat_df["CYP2C9 genotype unknown"] = wf.get_CYP2C9_genotype_unknown()
        self.feat_df["Asian race"] = wf.get_asian_race()
        self.feat_df["Black or African American"] = wf.get_black_or_african_american()
        self.feat_df["Missing or Mixed race"] = wf.get_missing_or_mixed_race()
        self.feat_df["Enzyme inducer status"] = wf.get_enzyme_inducer_status()
        self.feat_df["Amiodarone status"] = wf.get_amiodarone_status()
        self.feat_df["Weekly warfarin dose"] = wf.get_weekly_warfarin_dose()
        if (self.bin_weekly_dose):
            self.feat_df[self.out_column] = wf.get_binned_weekly_warfarin_dose()


