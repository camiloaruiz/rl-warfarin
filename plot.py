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
import seaborn as sns
import matplotlib.ticker

# import matplotlib; matplotlib.use('TkAgg')
#import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt




# python main.py --model fixed_dose 
name1 = "model=fixed_dose__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model wcda 
name2 ="model=wcda__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model wpda 
name3 ="model=wpda__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"

# python main.py --model fixed_dose --nan_val_0
name4 ="model=fixed_dose__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=True"
# python main.py --model wcda  --nan_val_0
name5 ="model=wcda__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=True"
# python main.py --model wpda  --nan_val_0
name6 ="model=wpda__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=True"


# python main.py --model UCBNet --bound_constant 1 --feature_group 0
name7 ="model=UCBNet__bin_weekly_dose=3__bound_constant=1.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model ThompsonNet --R 0.0005 --feature_group 0
name8 ="model=ThompsonNet__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model eGreedy --e_0 0.1 --e_scale 1.0  --feature_group 0
name9 ="model=eGreedy__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"

# python main.py --model UCBNet --bound_constant 1 --feature_group 0 --nan_val_0
name10 ="model=UCBNet__bin_weekly_dose=3__bound_constant=1.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=True"
# python main.py --model ThompsonNet --R 0.0005 --feature_group 0 --nan_val_0
name11 ="model=ThompsonNet__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=True"
# python main.py --model eGreedy --e_0 0.1 --e_scale 1.0  --feature_group 0 --nan_val_0
name12 ="model=eGreedy__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=True"

# python main.py --model UCBNet --bound_constant 1 --feature_group 1
name13 ="model=UCBNet__bin_weekly_dose=3__bound_constant=1.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=1__nan_val_0=False"
# python main.py --model ThompsonNet --R 0.0005 --feature_group 1
name14 ="model=ThompsonNet__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=1__nan_val_0=False"
# python main.py --model eGreedy --e_0 0.1 --e_scale 1.0  --feature_group 1
name15 ="model=eGreedy__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=1__nan_val_0=False"

# python main.py --model UCBNet --bound_constant 1 --feature_group 1 --nan_val_0
name16 ="model=UCBNet__bin_weekly_dose=3__bound_constant=1.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=1__nan_val_0=True"
# python main.py --model ThompsonNet --R 0.0005 --feature_group 1 --nan_val_0
name17 ="model=ThompsonNet__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=1__nan_val_0=True"
# python main.py --model eGreedy --e_0 0.1 --e_scale 1.0  --feature_group 1 --nan_val_0
name18 ="model=eGreedy__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=1__nan_val_0=True"





# python main.py --model UCBNet --bound_constant 0.5 --feature_group 0
name19 = "model=UCBNet__bin_weekly_dose=3__bound_constant=0.5__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model UCBNet --bound_constant 1 --feature_group 0
name20 = "model=UCBNet__bin_weekly_dose=3__bound_constant=1.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model UCBNet --bound_constant 1.5 --feature_group 0
name21 = "model=UCBNet__bin_weekly_dose=3__bound_constant=1.5__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model UCBNet --bound_constant 2 --feature_group 0
name22 = "model=UCBNet__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model UCBNet --bound_constant 2.5 --feature_group 0
name23 = "model=UCBNet__bin_weekly_dose=3__bound_constant=2.5__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"



# python main.py --model ThompsonNet --R 0.05 --feature_group 0
name24 = "model=ThompsonNet__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.05__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model ThompsonNet --R 0.005 --feature_group 0
name25 = "model=ThompsonNet__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model ThompsonNet --R 0.001 --feature_group 0
name26 = "model=ThompsonNet__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.001__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model ThompsonNet --R 0.0005 --feature_group 0
name27 = "model=ThompsonNet__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model ThompsonNet --R 0.0001 --feature_group 0
name28 = "model=ThompsonNet__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0001__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model ThompsonNet --R 0.00005 --feature_group 0
name29 = "model=ThompsonNet__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=5e-05__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"



# python main.py --model eGreedy --e_0 0.1 --e_scale 0  --feature_group 0
name30 = "model=eGreedy__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=0.0__feature_group=0__nan_val_0=False"
# python main.py --model eGreedy --e_0 0.1 --e_scale 0.5  --feature_group 0
name31 = "model=eGreedy__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=0.5__feature_group=0__nan_val_0=False"
# python main.py --model eGreedy --e_0 0.1 --e_scale 1.0  --feature_group 0
name32 = "model=eGreedy__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model eGreedy --e_0 0.1 --e_scale 2.0  --feature_group 0
name33 = "model=eGreedy__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=2.0__feature_group=0__nan_val_0=False"
# python main.py --model eGreedy --e_0 0.2 --e_scale 0.5  --feature_group 0
name34 = "model=eGreedy__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.2__e_scale=0.5__feature_group=0__nan_val_0=False"
# python main.py --model eGreedy --e_0 0.2 --e_scale 1.0  --feature_group 0
name35 = "model=eGreedy__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.2__e_scale=1.0__feature_group=0__nan_val_0=False"
# python main.py --model eGreedy --e_0 0.2 --e_scale 2.0  --feature_group 0
name36 = "model=eGreedy__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.2__e_scale=2.0__feature_group=0__nan_val_0=False"


# python main.py --model ThompsonDNet --R 0.0005 --feature_group 0
name37 = "model=ThompsonDNet__bin_weekly_dose=3__bound_constant=2.0__num_force=1__num_force_TH=0__R=0.0005__delta=0.1__epsilon=0.14476482730108395__num_trials=20__e_0=0.1__e_scale=1.0__feature_group=0__nan_val_0=False"


def plot_combined(scalar_results):
	points = defaultdict(list)
	for run_results in scalar_results:
		for step, value in enumerate(run_results):
			points[step].append(value)

	xs = sorted(points.keys())
	values = np.array([points[x] for x in xs])
	ys = np.mean(values, axis=1)
	
	## Prior with standard error of mean
	# yerrs = stats.sem(values, axis=1)
	# plt.fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.25)

	## Now with standard deviation
	yerrs = stats.tstd(values, limits = None, axis = 1)
	plt.fill_between(xs, ys - 1.96*yerrs, ys + 1.96*yerrs, alpha=0.25)
	
	plt.plot(xs, ys)
	

def plot_individually(run_results):
	xs = [step for step, value in run_results]
	ys = [value for step, value in run_results]
	plt.plot(xs, ys)

def plot(results_list, names, title,xlabel, ylabel, combine, figsize, extension, fontsize, x_major_tick_locator, x_minor_tick_locator, y_major_tick_locator, y_minor_tick_locator, major_tick_len, minor_tick_len, plots_dir="plot/"):
	fig, ax = plt.subplots(figsize = figsize)
	# Plot Data
	for results in results_list:
		if combine:
			plot_combined(results)
		else:
			plot_individually(results)

	# Title, Axes, Tick Marks, Legend
	plt.title(title, fontsize = fontsize)
	plt.xlabel(xlabel, fontsize = fontsize)
	plt.ylabel(ylabel, fontsize = fontsize)	
	plt.tick_params(axis='both', which='major', labelsize=fontsize)
	plt.tick_params(axis='both', which='minor', labelsize=fontsize)
	plt.legend(names, fontsize = fontsize)

	# Despine, Add Minor Tick Marks
	sns.despine(right = True, top = True)
	ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(x_major_tick_locator))
	ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(x_minor_tick_locator))
	ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(y_major_tick_locator))
	ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(y_minor_tick_locator))
	ax.tick_params(which = 'major', length = major_tick_len)
	ax.tick_params(which = 'minor', length = minor_tick_len)

	# Save
	suffix = '_combined' if combine else '_individual'
	save_path = plots_dir+title.replace(" ","_") + suffix + extension
	plt.savefig(str(save_path), bbox_inches = "tight")

def load_(name):
	all_a_star_a_hat = np.load("data/"+ name +"__a_star_a_hat.npy")
	all_regret_expected = np.load("data/"+ name +"__regret_expected.npy")
	all_regret_observed = np.load("data/"+ name +"__regret_observed.npy")
	all_frac_incorrect = np.load("data/"+ name +"__frac_incorrect.npy")
	all_frac_correct  = np.load("data/"+ name +"__frac_correct.npy")

	return all_a_star_a_hat, all_frac_incorrect, all_frac_correct, all_regret_expected, all_regret_observed


if __name__ == "__main__":
	# Plotting Parameters
	plot_params = {"figsize": (5, 5), "extension": ".pdf", "fontsize": 8, "expected_regret_big": {"y_major_tick_locator": 50, "y_minor_tick_locator": 10}, "expected_regret": {"y_major_tick_locator": 0.1, "y_minor_tick_locator": 0.02}, "oracle_regret": {"y_major_tick_locator": 100, "y_minor_tick_locator": 20}, "frac_incorrect": {"y_major_tick_locator": 0.1, "y_minor_tick_locator": 0.02}, "patient": {"x_major_tick_locator": 500, "x_minor_tick_locator": 100}, "major_tick_len": 7, "minor_tick_len": 3}

	# Initial Plots
	## Baseline NaN Removed
	data = []
	names = ["fixed_dose","wcda","wpda"]
	for name in [name1, name2, name3]:
		data.append(load_(name)[2])
	plot(results_list=data, names=names, title="Baseline NaN removed", xlabel="patient", ylabel="Frac Correct", combine=True, figsize = plot_params["figsize"], extension = plot_params["extension"], fontsize = plot_params["fontsize"], x_major_tick_locator = plot_params["patient"]["x_major_tick_locator"], x_minor_tick_locator = plot_params["patient"]["x_minor_tick_locator"], y_major_tick_locator = plot_params["frac_incorrect"]["y_major_tick_locator"], y_minor_tick_locator = plot_params["frac_incorrect"]["y_minor_tick_locator"], major_tick_len = plot_params["major_tick_len"], minor_tick_len = plot_params["minor_tick_len"])

	## Baseline NaN Replaced with 0
	data = []
	names = ["fixed_dose","wcda","wpda"]
	for name in [name4, name5, name6]:
		data.append(load_(name)[2])
	plot(results_list=data, names=names, title="Baseline NaN replaced with 0", xlabel="patient", ylabel="Frac Correct", combine=True, figsize = plot_params["figsize"], extension = plot_params["extension"], fontsize = plot_params["fontsize"], x_major_tick_locator = plot_params["patient"]["x_major_tick_locator"], x_minor_tick_locator = plot_params["patient"]["x_minor_tick_locator"], y_major_tick_locator = plot_params["frac_incorrect"]["y_major_tick_locator"], y_minor_tick_locator = plot_params["frac_incorrect"]["y_minor_tick_locator"], major_tick_len = plot_params["major_tick_len"], minor_tick_len = plot_params["minor_tick_len"])

	## Baseline NaN Removed; WPDA Features
	data = []
	names = ["fixed_dose","wcda","wpda","UCBNet","ThompsonNet","eGreedy"]
	for name in [name1, name2, name3, name7, name8, name9,]:
		data.append(load_(name)[1])
	plot(results_list=data, names=names, title="Baseline NaN removed; wpda features", xlabel="patient", ylabel="Frac Incorrect", combine=True, figsize = plot_params["figsize"], extension = plot_params["extension"], fontsize = plot_params["fontsize"], x_major_tick_locator = plot_params["patient"]["x_major_tick_locator"], x_minor_tick_locator = plot_params["patient"]["x_minor_tick_locator"], y_major_tick_locator = plot_params["frac_incorrect"]["y_major_tick_locator"], y_minor_tick_locator = plot_params["frac_incorrect"]["y_minor_tick_locator"], major_tick_len = plot_params["major_tick_len"], minor_tick_len = plot_params["minor_tick_len"])

	## Baseline Nan Replaced with 0; WPDA Features
	data = []
	names = ["fixed_dose","wcda","wpda","UCBNet","ThompsonNet","eGreedy"]
	for name in [name4, name5, name6, name10, name11, name12]:
		data.append(load_(name)[1])
	plot(results_list=data, names=names, title="Baseline NaN replaced with 0; wpda features", xlabel="patient", ylabel="Frac Incorrect", combine=True, figsize = plot_params["figsize"], extension = plot_params["extension"], fontsize = plot_params["fontsize"], x_major_tick_locator = plot_params["patient"]["x_major_tick_locator"], x_minor_tick_locator = plot_params["patient"]["x_minor_tick_locator"], y_major_tick_locator = plot_params["frac_incorrect"]["y_major_tick_locator"], y_minor_tick_locator = plot_params["frac_incorrect"]["y_minor_tick_locator"], major_tick_len = plot_params["major_tick_len"], minor_tick_len = plot_params["minor_tick_len"])

	## Baseline NaN Removed; All Features
	data = []
	names = ["fixed_dose","wcda","wpda","UCBNet","ThompsonNet","eGreedy"]
	for name in [name1, name2, name3, name13, name14, name15]:
		data.append(load_(name)[1])
	plot(results_list=data, names=names, title="Baseline NaN removed; All features", xlabel="patient", ylabel="Frac Incorrect", combine=True, figsize = plot_params["figsize"], extension = plot_params["extension"], fontsize = plot_params["fontsize"], x_major_tick_locator = plot_params["patient"]["x_major_tick_locator"], x_minor_tick_locator = plot_params["patient"]["x_minor_tick_locator"], y_major_tick_locator = plot_params["frac_incorrect"]["y_major_tick_locator"], y_minor_tick_locator = plot_params["frac_incorrect"]["y_minor_tick_locator"], major_tick_len = plot_params["major_tick_len"], minor_tick_len = plot_params["minor_tick_len"])

	## Baseline NaN Replaced; All Features
	data = []
	names = ["fixed_dose","wcda","wpda","UCBNet","ThompsonNet","eGreedy"]
	for name in [name4, name5, name6, name16, name17, name18]:
		data.append(load_(name)[1])
	plot(results_list=data, names=names, title="Baseline NaN replaced with; All features", xlabel="patient", ylabel="Frac Incorrect", combine=True, figsize = plot_params["figsize"], extension = plot_params["extension"], fontsize = plot_params["fontsize"], x_major_tick_locator = plot_params["patient"]["x_major_tick_locator"], x_minor_tick_locator = plot_params["patient"]["x_minor_tick_locator"], y_major_tick_locator = plot_params["frac_incorrect"]["y_major_tick_locator"], y_minor_tick_locator = plot_params["frac_incorrect"]["y_minor_tick_locator"], major_tick_len = plot_params["major_tick_len"], minor_tick_len = plot_params["minor_tick_len"])

	## Baseline NaN removed; WPDA features; Expected Regret
	data = []
	names = ["fixed_dose","wcda","wpda","UCBNet","ThompsonNet","eGreedy"]
	for name in [name1, name2, name3, name7, name8, name9,]:
		data.append(load_(name)[3])
	plot(results_list=data, names=names, title="Expected Regret NaN removed; wpda features", xlabel="patient", ylabel="Expected Regret", combine=True, figsize = plot_params["figsize"], extension = plot_params["extension"], fontsize = plot_params["fontsize"], x_major_tick_locator = plot_params["patient"]["x_major_tick_locator"], x_minor_tick_locator = plot_params["patient"]["x_minor_tick_locator"], y_major_tick_locator = plot_params["expected_regret_big"]["y_major_tick_locator"], y_minor_tick_locator = plot_params["expected_regret_big"]["y_minor_tick_locator"], major_tick_len = plot_params["major_tick_len"], minor_tick_len = plot_params["minor_tick_len"])

	## Baseline NaN removed; WPDA features; Oracle Regret
	data = []
	names = ["fixed_dose","wcda","wpda","UCBNet","ThompsonNet","eGreedy"]
	for name in [name1, name2, name3, name7, name8, name9,]:
		data.append(load_(name)[4])
	plot(results_list=data, names=names, title="Oracle Regret NaN removed; wpda features", xlabel="patient", ylabel="Oracle Regret", combine=True, figsize = plot_params["figsize"], extension = plot_params["extension"], fontsize = plot_params["fontsize"], x_major_tick_locator = plot_params["patient"]["x_major_tick_locator"], x_minor_tick_locator = plot_params["patient"]["x_minor_tick_locator"], y_major_tick_locator = plot_params["oracle_regret"]["y_major_tick_locator"], y_minor_tick_locator = plot_params["oracle_regret"]["y_minor_tick_locator"], major_tick_len = plot_params["major_tick_len"], minor_tick_len = plot_params["minor_tick_len"])


	# Hyperparameter sensitivity
	## Linear Alpha Search
	data = []
	names = ["0.5","1.0","1.5","2.0","2.5"]
	for name in [name19, name20, name21, name22, name23]:
		data.append(load_(name)[1])
	plot(results_list=data, names=names, title="Linear alpha search", xlabel="patient", ylabel="Frac Incorrect", combine=True, figsize = plot_params["figsize"], extension = plot_params["extension"], fontsize = plot_params["fontsize"], x_major_tick_locator = plot_params["patient"]["x_major_tick_locator"], x_minor_tick_locator = plot_params["patient"]["x_minor_tick_locator"], y_major_tick_locator = plot_params["frac_incorrect"]["y_major_tick_locator"], y_minor_tick_locator = plot_params["frac_incorrect"]["y_minor_tick_locator"], major_tick_len = plot_params["major_tick_len"], minor_tick_len = plot_params["minor_tick_len"])

	## Thompson R Search
	data = []
	names = ["0.005","0.001","0.0005","0.0001","0.00005"]
	for name in [name25, name26, name27, name28, name29]:
		data.append(load_(name)[1])
	plot(results_list=data, names=names, title="Thompson R search", xlabel="patient", ylabel="Expected Regret", combine=True, figsize = plot_params["figsize"], extension = plot_params["extension"], fontsize = plot_params["fontsize"], x_major_tick_locator = plot_params["patient"]["x_major_tick_locator"], x_minor_tick_locator = plot_params["patient"]["x_minor_tick_locator"], y_major_tick_locator = plot_params["expected_regret"]["y_major_tick_locator"], y_minor_tick_locator = plot_params["expected_regret"]["y_minor_tick_locator"], major_tick_len = plot_params["major_tick_len"], minor_tick_len = plot_params["minor_tick_len"])

	## eGreedy E and Decay Rate Search
	data = []
	names = ["0.1, 0", "0.1, 0.5", "0.1, 1", "0.1, 2", "0.2, 0.5", "0.2, 1","0.2, 2"]
	for name in [name30, name31, name32, name33, name34, name35, name36]:
		data.append(load_(name)[1])
	plot(results_list=data, names=names, title="eGreedy E and decay rate search", xlabel="patient", ylabel="Expected Regret", combine=True, figsize = plot_params["figsize"], extension = plot_params["extension"], fontsize = plot_params["fontsize"], x_major_tick_locator = plot_params["patient"]["x_major_tick_locator"], x_minor_tick_locator = plot_params["patient"]["x_minor_tick_locator"], y_major_tick_locator = plot_params["expected_regret"]["y_major_tick_locator"], y_minor_tick_locator = plot_params["expected_regret"]["y_minor_tick_locator"], major_tick_len = plot_params["major_tick_len"], minor_tick_len = plot_params["minor_tick_len"])