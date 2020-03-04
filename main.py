import argparse
from model.wpda import WPDA
from model.wcda import WCDA
from model.fixed_dose import FixedDose
from loader.warfarin_loader import WarfarinLoader
from evaluation.evaluation import Evaluation
import numpy as np

from model.UCBLin import UCBNET
import torch.optim as optim

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
        y_hat = model.predict(x)
        Y_hat = np.append(Y_hat, y_hat)
        evaluation_obj.evaluate(Y_hat, Y[:i])

    if False:
		net = UCBNet(bin_weekly_dose=args.bin_weekly_dose, num_actions=3, bound_constant=2)
		criterion = nn.MSELoss()
		optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    	for x, y, i in zip(X, Y, range(1, X.shape[0]+1)):
	        # zero the parameter gradients
	        optimizer.zero_grad()

	        # forward + backward + optimize
	        max_, i = net(torch.tensor([x],dtype=torch.float),t)
	        if i == y:
	        	loss = criterion(max_, torch.tensor([0],dtype=torch.float))
	        else: 
	        	loss = criterion(max_, torch.tensor([-1],dtype=torch.float))
	        loss.backward()
	        optimizer.step()
	        Y_hat = np.append(Y_hat, i)
	        evaluation_obj.evaluate(Y_hat, Y[:i])


    print("frac_incorrect: " + str(evaluation_obj.get_frac_incorrect()[-1]))



