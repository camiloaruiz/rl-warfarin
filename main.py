import argparse
from model.wpda import WPDA
from model.wcda import WCDA
from model.fixed_dose import FixedDose
from loader.warfarin_loader import WarfarinLoader
from evaluation.evaluation import Evaluation
import numpy as np

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
    parser.add_argument('--model', type = str, choices = ["fixed_dose", "wpda", "wcda"])
    parser.add_argument('--bin_weekly_dose', type=str2bool, nargs = "?", const=True)
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
    else:
        assert(False)
    
    # Prepare data
    model.featurize(wf)
    model.prepare_XY()
    X = model.get_X()
    Y = model.get_Y()

    # Run the model and evaluate
    evaluation_obj = Evaluation(["frac_incorrect"])
    Y_hat = np.array([])
    for x, y, i in zip(X, Y, range(1, X.shape[0]+1)):
        y_hat = model.predict(y)
        Y_hat = np.append(Y_hat, y_hat)
        evaluation_obj.evaluate(Y_hat, Y[:i])

    print("frac_incorrect: " + str(evaluation_obj.get_frac_incorrect()[-1]))