import pickle

from sklearn.metrics import balanced_accuracy_score, classification_report


def accuracy(output, target, topk=(1,)):
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
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def compute_metrics(logits, target, out_fn=None, verbose=False):
    pred = logits.argmax(dim=1)
    if out_fn is not None:
        with open(out_fn, 'wb') as f:
            pickle.dump({'logits': logits, 'pred': pred, 'target': target}, f)
            if verbose:
                print(f"Stored test predictions in {out_fn}.")

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
