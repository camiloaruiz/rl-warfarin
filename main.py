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

from collections import defaultdict
import scipy.stats as stats

import matplotlib; matplotlib.use('TkAgg')
#import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt





def plot_combined(scalar_results):
	points = defaultdict(list)
	for run_results in scalar_results:
		for step, value in run_results:
			points[step].append(value)
	
	xs = sorted(points.keys())
	values = np.array([points[x] for x in xs])
	ys = np.mean(values, axis=1)
	yerrs = stats.sem(values, axis=1)
	plt.fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.25)
	plt.plot(xs, ys)
	
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
	parser.add_argument('--model', type = str, choices = ["fixed_dose", "wpda", "wcda","UCBNet", "UCBDNet", "ThompsonNet", "ThompsonDNet"], required=True)
	parser.add_argument('--bin_weekly_dose', type=str2bool, nargs = "?", const=True)
	parser.add_argument('--bound_constant', type=float, nargs = "?", default=2.0)
	parser.add_argument('--num_force', type=int, nargs = "?", default=1)
	parser.add_argument('--R', type=float, nargs = "?", default=0.5)
	parser.add_argument('--delta', type=float, nargs = "?", default=0.1)
	parser.add_argument('--epsilon', type=float, nargs = "?", default=1.0/np.log(1000))
	parser.add_argument('--num_trials', type=int, nargs = "?", default=1)


	# These still need their corresponding use cases to be written 
	parser.add_argument('--num_bins', type=int, nargs = "?", const=3)
	parser.add_argument('--feature_group', type=int, nargs = "?", const=0)

	args = parser.parse_args()
	return args



if __name__ == "__main__":
	args = parse_args()

	# Get data
	wf = WarfarinLoader()


	# Instantiate model
	if args.model == "fixed_dose":
		model = FixedDose(args.bin_weekly_dose)
	elif args.model == "wpda":
		model = WPDA(args.bin_weekly_dose)
	elif args.model == "wcda":
		model = WCDA(args.bin_weekly_dose)
	elif args.model == "UCBNet":
		model = UCBNet(bin_weekly_dose=True, num_actions=3, bound_constant=2.0, num_force=1.0)
	elif args.model == "UCBDNet":
		model = UCBDNet(bin_weekly_dose=True, num_actions=3, bound_constant=2.0, num_force=1.0)
	elif args.model == "ThompsonNet":
		model = ThompsonNet(bin_weekly_dose=True, num_actions=3, R=0.5, delta=0.1, epsilon=1.0/np.log(1000), num_force=0.0)
	elif args.model == "ThompsonDNet":
		model = ThompsonDNet(bin_weekly_dose=True, num_actions=3, R=0.5, delta=0.1, epsilon=1.0/np.log(1000), num_force=0.0)
	else:
		assert(False)
	
	# Prepare data
	model.featurize(wf)
	model.prepare_XY()

	all_a_star_a_hat = []
	all_regret = []
	all_frac_incorrect = []
	for trial in range(args.num_trials):
		all_a_star_a_hat.append(model.experiment(rand_seed = trial)) 
		all_frac_incorrect.append(model.calc_frac_incorrect(all_a_star_a_hat[trial]))
		#all_regret.append(model.expected_regret(all_a_star_a_hat[tria]))

	for frac_incorrect in all_frac_incorrect:
		# plt.figure()
		plt.plot(range(1,len(frac_incorrect)+1), frac_incorrect)
		plt.show()














