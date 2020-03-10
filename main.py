import argparse
from model.wpda import WPDA
from model.wcda import WCDA
from model.fixed_dose import FixedDose
from loader.warfarin_loader import WarfarinLoader
import numpy as np

from model.UCBLin import UCBNet, UCBDNet
from model.ThompsonLin import ThompsonNet, ThompsonDNet
from model.eGreedy import eGreedy, eGreedyD

from collections import defaultdict
import scipy.stats as stats
from scipy.stats.stats import pearsonr
import sys


import matplotlib; matplotlib.use('TkAgg')
#import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




def plot_combined(scalar_results):
	points = defaultdict(list)
	for run_results in scalar_results:
		for step, value in enumerate(run_results):
			points[step].append(value)

	xs = sorted(points.keys())
	values = np.array([points[x] for x in xs])
	ys = np.mean(values, axis=1)
	yerrs = stats.sem(values, axis=1)
	plt.fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.25)
	plt.plot(xs, ys)
	plt.show()


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
	parser = argparse.ArgumentParser(description = "Runs experiment for warfarin dose prediction")
	parser.add_argument('--model', type = str, choices = ["fixed_dose", "wpda", "wcda","UCBNet", "UCBDNet", "ThompsonNet", "ThompsonDNet","eGreedy","eGreedyD"], required=True)
	parser.add_argument('--bin_weekly_dose', type=int,  choices =[1,2,3,4,5], default=3)
	parser.add_argument('--bound_constant', type=float, nargs = "?", default=2.0)
	parser.add_argument('--num_force', type=int, nargs = "?", default=1)
	parser.add_argument('--num_force_TH', type=int, nargs = "?", default=0)
	parser.add_argument('--R', type=float, nargs = "?", default=0.5)
	parser.add_argument('--delta', type=float, nargs = "?", default=0.1)
	parser.add_argument('--epsilon', type=float, nargs = "?", default=1.0/np.log(1000))
	parser.add_argument('--num_trials', type=int, nargs = "?", default=30)
	parser.add_argument('--e_0', type=float, nargs = "?", default=0.1)
	parser.add_argument('--e_scale', type=float, nargs = "?", default=1.0)
	parser.add_argument('--feature_group', type=int, nargs = "?", default=0)
	parser.add_argument("--nan_val_0", type=str2bool, nargs='?', const=True, default=False, help="Activate nan replaced to 0 mode.")


	args = parser.parse_args()
	return args

def get_model(args):
		# Instantiate model
	if args.model == "fixed_dose":
			model = FixedDose(bin_weekly_dose=args.bin_weekly_dose)
	elif args.model == "wpda":
			model = WPDA(bin_weekly_dose=args.bin_weekly_dose)
	elif args.model == "wcda":
			model = WCDA(bin_weekly_dose=args.bin_weekly_dose)
	elif args.model == "UCBNet":
			model = UCBNet(bin_weekly_dose=args.bin_weekly_dose, num_actions=args.bin_weekly_dose, bound_constant=args.bound_constant, num_force=args.num_force, feature_group=args.feature_group)
	elif args.model == "UCBDNet":
			model = UCBDNet(bin_weekly_dose=args.bin_weekly_dose, num_actions=args.bin_weekly_dose, bound_constant=args.bound_constant, num_force=args.num_force, feature_group=args.feature_group)
	elif args.model == "ThompsonNet":
			model = ThompsonNet(bin_weekly_dose=args.bin_weekly_dose, num_actions=args.bin_weekly_dose, R=args.R, delta=0.1, epsilon=1.0/np.log(1000), num_force=args.num_force_TH, feature_group=args.feature_group)
	elif args.model == "ThompsonDNet":
			model = ThompsonDNet(bin_weekly_dose=args.bin_weekly_dose, num_actions=args.bin_weekly_dose, R=args.R, delta=0.1, epsilon=1.0/np.log(1000), num_force=args.num_force_TH, feature_group=args.feature_group)
	elif args.model == "eGreedy":
			model = eGreedy(bin_weekly_dose=args.bin_weekly_dose, num_actions=args.bin_weekly_dose, e_0 = args.e_0, e_scale = args.e_scale, feature_group=args.feature_group)
	elif args.model == "eGreedyD":
			model = eGreedyD(bin_weekly_dose=args.bin_weekly_dose, num_actions=args.bin_weekly_dose, e_0 = args.e_0, e_scale = args.e_scale, feature_group=args.feature_group)
	else:
		assert(False)

	# Prepare data
	model.featurize(wf)
	model.prepare_XY()
	return model 



if __name__ == "__main__":
	args = parse_args()
	name = [str(arg)+"="+str(getattr(args, arg)) for arg in vars(args)]
	name ="__".join(name)

	# Get data
	if args.nan_val_0:
		wf = WarfarinLoader(na_val=0.0,fill_na_mean=False,stable_dose_only=True)
	else: 
		wf = WarfarinLoader(na_val=np.nan,fill_na_mean=False,stable_dose_only=True)

	
	all_a_star_a_hat = []
	all_regret_expected,all_regret_observed, all_frac_incorrect, all_frac_correct = [],[],[]
	for trial in range(args.num_trials):
		model = get_model(args)

		a_star_a_hat = model.experiment(rand_seed = trial)
		
		if args.bin_weekly_dose == 1:
			cum_regret = model.non_binned_regret(a_star_a_hat)
			regret_observed = cum_regret[:]
			cum_frac_incorrect = model.non_binned_calc_frac_incorrect(a_star_a_hat)
			frac_correct = []
		else:
			cum_frac_incorrect = model.calc_frac_incorrect(a_star_a_hat)
			cum_regret = model.expected_regret(a_star_a_hat)		
			regret_observed = model.observed_regret(a_star_a_hat)
			frac_correct = model.calc_frac_correct(a_star_a_hat)


		all_a_star_a_hat.append(a_star_a_hat)
		all_frac_incorrect.append(cum_frac_incorrect)
		all_frac_correct.append(frac_correct)
		all_regret_expected.append(cum_regret)
		all_regret_observed.append(regret_observed)

	
	avg_frac_incorrect = np.mean(np.array(all_frac_incorrect), axis=0)
	avg_regret_expected = np.mean(np.array(all_regret_expected), axis=0)
	avg_regret_observed = np.mean(np.array(all_regret_observed), axis=0)
	
	print (args.model, "Averaged Frac-Incorrect / Final Regret expected / Final Regret observed ", avg_frac_incorrect[-1], avg_regret_expected[-1], avg_regret_observed[-1])
	plot_combined(all_frac_incorrect)
	plot_combined(all_regret_expected)
	plot_combined(all_regret_observed)



	#Code for adams plotting functions
	np.save("data/"+ name+"__a_star_a_hat",all_a_star_a_hat)
	np.save("data/"+ name+"__regret_expected",all_regret_expected)
	np.save("data/"+ name+"__regret_observed",all_regret_observed)
	np.save("data/"+ name+"__frac_incorrect",all_frac_incorrect)
	np.save("data/"+ name+"__frac_correct",all_frac_correct)


	#plot_combined(all_frac_incorrect)

	"""
	a_star_a_hat = model.experiment(rand_seed = 7)
	#print(a_star_a_hat)
	frac_incorrect = model.calc_final_frac_incorrect(a_star_a_hat)
	print(args.model,"Frac Incorrect= ",frac_incorrect)
	plt.plot(model.regret_over_time(a_star_a_hat))
	plt.xlabel("Sample")
	plt.ylabel("Cumulative Regret")
	plt.show()
	"""












