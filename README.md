# How flawed is ECE?

Code for the paper: https://arxiv.org/abs/2402.10046.

# Recreating Experiments

## Synthetic and CIFAR Experiments
All the synthetic and CIFAR-10 plots from the paper can be recreated by running the notebook `notebooks/cifar_experiments.ipynb`. Running this notebook requires updating the first cell to point to your project directory.

## ImageNet Experiments
All ImageNet experiments can be recreated by running `python3 run_imagenet_eval.py` followed by `python3 generate_imagenet_plots.py`. Warning: the former command will cache several large models and compute and store their logits over ImageNet-1K-Val. 
