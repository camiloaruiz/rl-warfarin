import argparse
from model.wpda import WPDA
from model.wcda import WCDA
from model.fixed_dose import FixedDose
from loader.warfarin_loader import WarfarinLoader
import numpy as np

from model.UCBLin import UCBNet
from model.UCBLinD import UCBDNet
from model.ThompsonLin import ThompsonNet
from model.ThompsonD import ThompsonDNet
from model.eGreedy import eGreedy

from collections import defaultdict
import scipy.stats as stats

import matplotlib; matplotlib.use('TkAgg')
#import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt





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

def plot_individually(run_results):
	xs = [step for step, value in run_results]
	ys = [value for step, value in run_results]
	plt.plot(xs, ys)

def plot(results_list, names, title, combine, plots_dir):
	plt.figure()
	plt.title(title)
	plt.xlabel('Step')
	for results in results_list:
		if combine:
			plot_combined(results)
		else:
			plot_individually(results)
	suffix = '_combined' if combine else '_individual'
	save_path = plots_dir / (title + suffix + '.png')
	plt.legend(names)
	plt.savefig(str(save_path))


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
	parser.add_argument('--model', type = str, choices = ["fixed_dose", "wpda", "wcda","UCBNet", "UCBDNet", "ThompsonNet", "ThompsonDNet","eGreedy"], required=True)
	parser.add_argument('--bin_weekly_dose', type=str2bool, nargs = "?", const=True)
	parser.add_argument('--bound_constant', type=float, nargs = "?", default=2.0)
	parser.add_argument('--num_force', type=int, nargs = "?", default=1)
	parser.add_argument('--num_force_TH', type=int, nargs = "?", default=0)
	parser.add_argument('--R', type=float, nargs = "?", default=0.5)
	parser.add_argument('--delta', type=float, nargs = "?", default=0.1)
	parser.add_argument('--epsilon', type=float, nargs = "?", default=1.0/np.log(1000))
	parser.add_argument('--num_trials', type=int, nargs = "?", default=30)
	parser.add_argument('--e_0', type=float, nargs = "?", default=0.1)


	# These still need their corresponding use cases to be written
	parser.add_argument('--num_bins', type=int, nargs = "?", const=3)
	parser.add_argument('--feature_group', type=int, nargs = "?", const=0)

	args = parser.parse_args()
	return args



if __name__ == "__main__":
	args = parse_args()

	# Get data
	wf = WarfarinLoader(na_val=np.nan,fill_na_mean=False,stable_dose_only=False)


	# Instantiate model
	if args.model == "fixed_dose":
			model = FixedDose(bin_weekly_dose=True)
	elif args.model == "wpda":
			model = WPDA(bin_weekly_dose=True)
	elif args.model == "wcda":
			model = WCDA(bin_weekly_dose=True)
	elif args.model == "UCBNet":
			model = UCBNet(bin_weekly_dose=True, num_actions=3, bound_constant=args.bound_constant, num_force=args.num_force)
	elif args.model == "UCBDNet":
			model = UCBDNet(bin_weekly_dose=True, num_actions=3, bound_constant=args.bound_constant, num_force=args.num_force)
	elif args.model == "ThompsonNet":
			model = ThompsonNet(bin_weekly_dose=True, num_actions=3, R=args.R, delta=0.1, epsilon=1.0/np.log(1000), num_force=args.num_force_TH)
	elif args.model == "ThompsonDNet":
			model = ThompsonDNet(bin_weekly_dose=True, num_actions=3, R=args.R, delta=0.1, epsilon=1.0/np.log(1000), num_force=args.num_force_TH)
	elif args.model == "eGreedy":
			model = eGreedy(bin_weekly_dose=True, num_actions=3, e_0 = args.e_0)

	else:
		assert(False)

	# Prepare data
	model.featurize(wf)
	model.prepare_XY()

	all_regret = []
	all_a_star_a_hat = []
	all_frac_incorrect = []
	for trial in range(args.num_trials):
		all_a_star_a_hat.append(model.experiment(rand_seed = trial))
		frac_incorrect = model.calc_final_frac_incorrect(all_a_star_a_hat[-1])
		#if trial==0: print(args.model,"Frac Incorrect= ",frac_incorrect)
		all_frac_incorrect.append(model.calc_frac_incorrect(all_a_star_a_hat[trial]))
		#all_regret.append(model.regret_over_time(all_a_star_a_hat[trial]))
	print (args.model, "Averaged Frac Incorrect: ", np.mean(all_frac_incorrect))




	#Code for adams plotting functions
	#np.save("data/"+ args.model+"_regret",all_regret)
	#np.save("data/"+ args.model+"_frac_incorrect",all_frac_incorrect)

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












