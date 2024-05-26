import numpy as np
import torch
import datasets

from typing import Any, Tuple
from utils.ece_utils import *


def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> float:
    """Get test accuracy.

    Args:
        model (torch.nn.Module): Model.
        test_loader (torch.utils.data.DataLoader): Data loader for test data.
        device (str, optional): Model/data device. Defaults to "cpu".

    Returns:
        float: Test accuracy.
    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    return 100 * (correct / len(test_loader.dataset))


def get_softmax_and_labels(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Get softmaxes and labels.

    Args:
        model (torch.nn.Module): Model.
        test_loader (torch.utils.data.DataLoader): Data loader for test data.
        device (str, optional): Model/data device. Defaults to "cpu".

    Returns:
        Tuple[np.ndarray, np.ndarray]: Softmaxes and labels.
    """
    model.eval()
    softmaxes, labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            softmaxes.append(torch.nn.functional.softmax(output, dim=1))
            labels.append(target)
    softmaxes = torch.cat(softmaxes)
    labels = torch.cat(labels)

    return softmaxes.cpu().numpy(), labels.cpu().numpy()


def get_binary_logits_and_labels(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Get binary logits and binary (correctness) labels.

    Args:
        model (torch.nn.Module): Model.
        test_loader (torch.utils.data.DataLoader): Data loader for test data.
        device (str, optional): Model/data device. Defaults to "cpu".

    Returns:
        Tuple[torch.FloatTensor, torch.LongTensor]: Binary logits, and binary correctness labels.
    """
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            bin_logits = inv_sigmoid(
                torch.nn.functional.softmax(output, dim=1)
                .max(dim=1, keepdim=True)
                .values
                - 1e-6
            )
            pred = output.argmax(dim=1, keepdim=True)
            bin_labels = pred.eq(target.view_as(pred))

            logits.append(bin_logits)
            labels.append(bin_labels)
    logits = torch.cat(logits)
    labels = torch.cat(labels)

    return logits.cpu(), labels.cpu().long()


def get_probs_logits_labels_stream(
    dataset: datasets.IterableDataset,
    model: torch.nn.Module,
    transforms: Any,
    cutoff: int = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, torch.FloatTensor, np.ndarray, torch.LongTensor]:
    """_summary_

    Args:
        dataset (datasets.IterableDataset): HuggingFace dataset.
        model (torch.nn.Module): Model.
        transforms (Any): Transforms for dataset.
        cutoff (int, optional): Cutoff for dataset. Defaults to None.
        device (str, optional): Device. Defaults to "cpu".

    Returns:
        Tuple[np.ndarray, torch.FloatTensor, np.ndarray, torch.LongTensor]: Softmaxes, binary logits, original labels, and binary (accuracy) labels.
    """
    model.to(device)
    model.eval()
    softmaxes, logits, labels, bin_labels = [], [], [], []
    seen = 0
    with torch.no_grad():
        for data in dataset:
            img = data["jpg"].convert("RGB")
            x, target = transforms(img).unsqueeze(0).to(device), data["cls"]
            output = model(x)

            softmax = torch.nn.functional.softmax(output, dim=1)
            bin_logits = inv_sigmoid(softmax.max(dim=1, keepdim=True).values - 1e-6)
            pred = output.argmax(dim=1).item()
            bin_label = int(pred == target)

            softmaxes.append(softmax)
            logits.append(bin_logits)
            labels.append(target)
            bin_labels.append(bin_label)
            seen += 1
            if cutoff is not None and seen >= cutoff:
                break

    softmaxes = torch.cat(softmaxes)
    logits = torch.cat(logits)
    labels = torch.LongTensor(labels)
    bin_labels = torch.LongTensor(bin_labels)

    return (
        softmaxes.cpu().numpy(),
        logits.cpu(),
        labels.cpu().numpy(),
        bin_labels.unsqueeze(dim=1).cpu(),
    )
