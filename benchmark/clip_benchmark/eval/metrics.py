from typing import List

import torch
from sklearn.metrics import balanced_accuracy_score, classification_report


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> List[float]:
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.

    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.

    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies

    Returns
    -------

    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(k=max(topk), dim=1, largest=True, sorted=True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def compute_metrics(logits: torch.Tensor, target: torch.Tensor, verbose: bool = False) -> dict:
    pred = logits.argmax(dim=1)

    # measure accuracy
    if target.max() >= 5:
        acc1, acc5 = accuracy(logits.float(), target.float(), topk=(1, 5))
    else:
        acc1, = accuracy(logits.float(), target.float(), topk=(1,))
        acc5 = float("nan")
    mean_per_class_recall = balanced_accuracy_score(target, pred)
    fair_info = {
        "acc1": acc1,
        "acc5": acc5,
        "mean_per_class_recall": mean_per_class_recall,
        "classification_report": classification_report(target, pred, digits=3)
    }
    if verbose:
        print(fair_info["classification_report"])
        print(f"Test acc1: {acc1}")

    return {"lp_acc1": fair_info["acc1"], "lp_acc5": fair_info["acc5"],
            "lp_mean_per_class_recall": fair_info["mean_per_class_recall"]}
