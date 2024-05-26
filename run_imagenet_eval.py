import argparse
import os
import pickle
import random
import sys

sys.path.append(os.getcwd())

import numpy as np
import relplot as rp
import timm
import torch

from datasets import load_dataset
from huggingface_hub import login
from netcal.metrics import ECE
from pathlib import Path

from utils.ece_utils import *
from utils.eval_utils import *
from utils.noise_utils import *
from utils.plotting_utils import *

parser = argparse.ArgumentParser(description="Noise distribution.")
parser.add_argument("--noise", dest="noise", default="gaussian", type=str)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device != "cpu":
    print("Device count: ", torch.cuda.device_count())
    print("GPU being used: {}".format(torch.cuda.get_device_name(0)))


seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Huggingface read credentials.
login(token=os.environ["HF_READ"])


# For getting eces.
def eval_model_eces(softmaxes, logits, labels, bin_labels, bin_var_range, n_t):
    bin_eces, ls_eces = [], []
    for n_bins in bin_var_range:
        ece = ECE(n_bins)
        bin_eces.append(ece.measure(softmaxes, labels))

        if args.noise.lower() == "gaussian":
            noise = GaussianNoise(sigma=1/n_bins)
        else:
            noise = UniformNoise(sigma=1/n_bins)
        ls_eces.append(logit_smoothed_ece(logits, bin_labels, n_t, noise))
    return bin_eces, ls_eces


# For ece computations.
n_t = 10000
bin_var_range = list(range(0, 101, 10))
bin_var_range[0] = 1

dataset = load_dataset("timm/imagenet-1k-wds", split="validation", streaming=True)
model_names = [
    "resnet18.a1_in1k",
    "resnet50.a1_in1k",
    "efficientnet_b0.ra_in1k",
    "mobilenetv3_large_100.ra_in1k",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "regnety_080.ra3_in1k",
]
for model_name in model_names:
    # Model path for storing results.
    model_path = f"results/{model_name}"
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # Model config.
    model = timm.create_model(model_name, pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Compute and store results.
    if Path(f"{model_path}/softmaxes.p").is_file():
        softmaxes = pickle.load(open(f"{model_path}/softmaxes.p", "rb"))
        bin_logits = pickle.load(open(f"{model_path}/bin_logits.p", "rb"))
        labels = pickle.load(open(f"{model_path}/labels.p", "rb"))
        bin_labels = pickle.load(open(f"{model_path}/bin_labels.p", "rb"))
    else:
        softmaxes, bin_logits, labels, bin_labels = get_probs_logits_labels_stream(
            dataset, model, transforms, cutoff=None, device=device
        )
        pickle.dump(softmaxes, open(f"{model_path}/softmaxes.p", "wb"))
        pickle.dump(bin_logits, open(f"{model_path}/bin_logits.p", "wb"))
        pickle.dump(labels, open(f"{model_path}/labels.p", "wb"))
        pickle.dump(bin_labels, open(f"{model_path}/bin_labels.p", "wb"))
    
    bin_eces, ls_eces = eval_model_eces(
        softmaxes, bin_logits, labels, bin_labels, bin_var_range, n_t
    )
    print(f"LS-ECES: {ls_eces}")
    max_probs = softmaxes.max(axis=1)
    sm_ece = np.array(rp.smECE(max_probs, np.squeeze(bin_labels.numpy(), axis=1)))
    print(f"smECE: {sm_ece}")

    pickle.dump(bin_var_range, open(f"{model_path}/bin_var_range.p", "wb"))
    pickle.dump(bin_eces, open(f"{model_path}/bin_eces.p", "wb"))
    pickle.dump(ls_eces, open(f"{model_path}/ls_eces_{args.noise.lower()}.p", "wb"))
    pickle.dump(sm_ece, open(f"{model_path}/sm_ece.p", "wb"))
    print(f"Finished {model_name}.\n")
