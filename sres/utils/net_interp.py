""" Network Interpolation """
import argparse
import torch
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--alpha',
                    type=float,
                    required=False,
                    default=0.3)
args = parser.parse_args()
alpha = args.alpha

def net_interp(netA_path, netB_path, net_interp_path, alpha=0.3):

    netA = torch.load(netA_path)
    netB = torch.load(netB_path)
    net_interp = OrderedDict()

    print('Interpolating with alpha = ', alpha)

    for k, v_netA in netA.items():
        v_netB = netB[k]
        net_interp[k] = (1 - alpha) * v_netA + alpha * v_netB

    torch.save(net_interp, net_interp_path)