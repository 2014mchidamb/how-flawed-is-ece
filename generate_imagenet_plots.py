import argparse
import os
import pickle
import sys

import numpy as np

sys.path.append(os.getcwd())

from pathlib import Path
from utils.plotting_utils import *

parser = argparse.ArgumentParser(description="Noise distribution.")
parser.add_argument("--noise", dest="noise", default="gaussian", type=str)
args = parser.parse_args()

# Make sure plots directory exists.
Path("plots").mkdir(parents=True, exist_ok=True)

# Can also do this programmatically, but being lazy.
model_names = [
    "resnet18.a1_in1k",
    "resnet50.a1_in1k",
    "efficientnet_b0.ra_in1k",
    "mobilenetv3_large_100.ra_in1k",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "regnety_080.ra3_in1k",
]
bin_var_range = pickle.load(open(f"results/{model_names[0]}/bin_var_range.p", "rb"))
bin_eces, ls_eces, sm_eces = [], [], []
for i, model_name in enumerate(model_names):
    bin_eces.append(100 * np.array(pickle.load(open(f"results/{model_name}/bin_eces.p", "rb"))))
    ls_eces.append(100 * np.array(pickle.load(open(f"results/{model_name}/ls_eces_{args.noise.lower()}.p", "rb"))))
    sm_eces.append(100 * np.repeat(pickle.load(open(f"results/{model_name}/sm_ece.p", "rb")), len(ls_eces[-1])))
    # Generate individual plots along the way.
    plot_multi_dataset_metrics(
        fname=f"plots/{model_name.split('.')[0]}_imagenet_{args.noise.lower()}.png",
        x_label=r"Number of Bins $(1/\sigma)$",
        y_label=f"{model_name.split('.')[0]} ECE (%)",
        xs=bin_var_range,
        metric_means=[bin_eces[-1], ls_eces[-1], sm_eces[-1]],
        metric_stds=None,
        datasets=["Binned ECE", "LS-ECE", "smECE"],
        custom_colors=[f"C{i+1}", f"C{i+1}", f"black"],
        custom_lines=["solid", "dashed", "solid"],
    )

# Generate combined plots.
ls_ece_diffs = np.abs(np.array(bin_eces) - np.array(ls_eces))
sm_ece_diffs = np.abs(np.array(bin_eces) - np.array(sm_eces))
diff_means = [ls_ece_diffs.mean(axis=0), sm_ece_diffs.mean(axis=0)]
diff_stds = [ls_ece_diffs.std(axis=0), sm_ece_diffs.std(axis=0)]

plot_multi_dataset_metrics(
    fname=f"plots/imagenet_eval_{args.noise.lower()}.png",
    x_label=r"Number of Bins $(1/\sigma)$",
    y_label="Mean Absolute Diference of ECEs (%)",
    xs=bin_var_range,
    metric_means=diff_means,
    metric_stds=diff_stds,
    datasets=["LS-ECE/ECE Difference", "smECE/ECE Difference"],
    y_lim=3.0,
)
