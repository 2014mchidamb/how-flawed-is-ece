import numpy as np
import torch


def inv_sigmoid(x: torch.FloatTensor) -> torch.FloatTensor:
    """Inverse sigmoid.

    Args:
        x (torch.FloatTensor): Torch tensor to apply to.

    Returns:
        torch.FloatTensor: x after applying inverse sigmoid.
    """
    return torch.log(x) - torch.log(1 - x)


def kernel_reg(logits: torch.FloatTensor, labels: torch.LongTensor, ts: torch.FloatTensor, noise) -> torch.FloatTensor:
    """Computes kernel regression using provided noise distribition.

    Args:
        logits (torch.FloatTensor): Logits tensor.
        labels (torch.LongTensor): Labels tensor.
        ts (torch.FloatTensor): Tensor of noised logits.
        noise: Should be either GaussianNoise or UniformNoise.

    Returns:
        torch.FloatTensor: Returns estimate of conditional expectation.
    """
    total = noise.kernel(ts.unsqueeze(dim=1) - logits)
    return (total * labels.unsqueeze(dim=0)).mean(dim=1) / total.mean(dim=1)


def logit_smoothed_ece(logits: torch.FloatTensor, labels: torch.LongTensor, n_t: int, noise, reduce: bool = True) -> float:
    """Computes logit smoothed ECE.

    Args:
        logits (torch.FloatTensor): Logits tensor.
        labels (torch.LongTensor): Labels tensor.
        ts (torch.FloatTensor): Tensor of noised logits.
        noise: Should be either GaussianNoise or UniformNoise.
        reduce (bool, optional): Whether to return reduced value or not. Defaults to True.

    Returns:
        float: Logit-smoothed ECE value, returned if reduce=True. 
    """
    # Expects logits to be shape (n, 1) and labels to be shape (n, 1).
    emp_sample = torch.randint(len(logits), (n_t,))
    ts = logits[emp_sample] + noise.sample((n_t, 1))
    ests = kernel_reg(logits, labels, ts, noise)
    if reduce:
        return torch.abs((ests - torch.nn.functional.sigmoid(ts))).mean()
    else:
        return torch.nn.functional.sigmoid(ts), ests