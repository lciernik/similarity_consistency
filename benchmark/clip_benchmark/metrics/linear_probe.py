import os
import pickle
import time
from contextlib import suppress

import numpy as np
import torch
from sklearn.metrics import classification_report, balanced_accuracy_score
from tqdm import tqdm

from .data_utils import Featurizer, feature_extraction, get_feature_dl, get_combined_feature_dl
from .feature_combiner import ConcatFeatureCombiner


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


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def train(dataloader,  weight_decay, lr, epochs, autocast, device, seed):
    torch.manual_seed(seed)
    input_shape, output_shape = dataloader.dataset[0][0].shape[0], dataloader.dataset[0][1].max().item() + 1
    model = torch.nn.Linear(input_shape, output_shape)
    devices = [x for x in range(torch.cuda.device_count())]
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=devices)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    len_loader = len(dataloader)
    scheduler = cosine_lr(optimizer, lr, 0., epochs * len_loader)

    for epoch in range(epochs):
        end = time.time()
        for i, (x, y) in enumerate(dataloader):
            x, y = x.cuda(), y.cuda()
            step = i + epoch * len_loader
            data_time = time.time() - end
            scheduler(step)

            optimizer.zero_grad()
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            if (i % 20) == 1:
                num_samples = i * len(x)
                try:
                    samples_per_epoch = len(dataloader)
                    percent_complete = 100.0 * i / len(dataloader)
                    progress_message = f"[{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)]"
                except TypeError:
                    progress_message = f"[{num_samples} samples]"
                print(
                    f"Train Epoch: {epoch} {progress_message}\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t"
                    f"LR {optimizer.param_groups[0]['lr']:.5f}"
                )
    return model


def infer(model, dataloader, autocast, device):
    true, pred = [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            y = y.to(device)

            with autocast():
                logits = model(x)

            pred.append(logits.cpu())
            true.append(y.cpu())

    logits = torch.cat(pred)
    target = torch.cat(true)
    return logits, target


def find_peak(wd_list, idxs, train_loader, val_loader, lr, epochs, autocast, device, verbose,
              seed):
    best_wd_idx, max_acc = 0, 0
    for idx in idxs:
        weight_decay = wd_list[idx]
        model = train(train_loader, weight_decay, lr, epochs, autocast, device, seed)
        logits, target = infer(model, val_loader, autocast, device)
        acc1, = accuracy(logits.float(), target.float(), topk=(1,))
        if verbose:
            print(f"Valid accuracy with weight_decay {weight_decay}: {acc1}")
        if max_acc < acc1:
            best_wd_idx, max_acc = idx, acc1
    return best_wd_idx

def tune_weight_decay(feature_train_loader, feature_val_loader,
                         lr, epochs, autocast, device, verbose, seed):
    # perform openAI-like hyperparameter sweep
    # https://arxiv.org/pdf/2103.00020.pdf A.3
    # instead of scikit-learn LBFGS use FCNNs with AdamW
    wd_list = np.logspace(-6, 2, num=97).tolist()
    wd_list_init = np.logspace(-6, 2, num=7).tolist()
    wd_init_idx = [i for i, val in enumerate(wd_list) if val in wd_list_init]
    peak_idx = find_peak(wd_list, wd_init_idx, feature_train_loader, feature_val_loader,
                         lr, epochs, autocast, device, verbose, seed)
    step_span = 8
    while step_span > 0:
        left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(wd_list) - 1)
        peak_idx = find_peak(wd_list, [left, peak_idx, right], feature_train_loader, feature_val_loader,
                              lr, epochs, autocast, device, verbose, seed)
        step_span //= 2
    best_wd = wd_list[peak_idx]
    return best_wd

def _evaluate(train_loader, best_wd, fewshot_k, feature_test_loader,
              lr, epochs, seed, device, autocast, out_fn=None, normalize=True, verbose=False):
    final_model = train(train_loader, best_wd, lr, epochs, autocast, device, seed)
    logits, target = infer(final_model, feature_test_loader, autocast, device)
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
        "weight_decay": best_wd,
        "acc1": acc1,
        "acc5": acc5,
        "mean_per_class_recall": mean_per_class_recall,
        "classification_report": classification_report(target, pred, digits=3)
    }
    if verbose:
        print(fair_info["classification_report"])
        print(f"Test acc1: {acc1} with weight_decay: {best_wd}")

    return {"lp_acc1": fair_info["acc1"], "lp_acc5": fair_info["acc5"],
            "lp_mean_per_class_recall": fair_info["mean_per_class_recall"],
            "weight_decay": fair_info['weight_decay'], 'epochs': epochs, 'seed': seed, 'fewshot_k': fewshot_k,
            'normalized': normalize}


def evaluate(model, train_dataloader, dataloader, fewshot_k, batch_size, num_workers, lr, epochs,
             model_id, seed, feature_root, device, val_dataloader=None, normalize=True, amp=True,
             out_fn=None, verbose=False):
    assert device == 'cuda'  # need to use cuda for this else too slow
    # first we need to featurize the dataset, and store the result in feature_root
    if not os.path.exists(feature_root):
        os.mkdir(feature_root)

    feature_dir = os.path.join(feature_root, model_id)
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)

    autocast = torch.cuda.amp.autocast if amp else suppress
    if not os.path.exists(os.path.join(feature_dir, 'targets_train.pt')):
        # We need to generate features if these do not exist
        featurizer = Featurizer(model, normalize).cuda()
        feature_extraction(featurizer, train_dataloader, dataloader, feature_dir, device, autocast, val_dataloader)

    feature_train_loader, feature_val_loader, feature_train_val_loader, feature_test_loader = get_feature_dl(
        feature_dir, batch_size, num_workers, fewshot_k, val_dataloader)



    if val_dataloader is not None:
        best_wd = tune_weight_decay(feature_train_loader, feature_val_loader,
                         lr, epochs, autocast, device, verbose, seed)
        train_loader = feature_train_val_loader
    else:
        best_wd = 0
        train_loader = feature_train_loader

    return _evaluate(train_loader=train_loader,
                     best_wd=best_wd,
                     fewshot_k=fewshot_k,
                     feature_test_loader=feature_test_loader,
                     lr=lr,
                     epochs=epochs,
                     seed=seed,
                     device=device,
                     autocast=autocast,
                     normalize=normalize,
                     out_fn=out_fn,
                     verbose=verbose)


def evaluate_combined(model_ids, feature_root, fewshot_k, batch_size, num_workers, lr, epochs, device, seed,
                      use_val_ds=False, amp=True, verbose=False, feature_combiner_cls=ConcatFeatureCombiner,
                      normalize=True, out_fn=None):
    assert device == 'cuda'

    assert os.path.exists(feature_root), "Feature root path non-existent"

    feature_dirs = [os.path.join(feature_root, model_id) for model_id in model_ids]
    print('feature_dirs', feature_dirs, flush=True)
    assert all([os.path.exists(feature_dir) for feature_dir in feature_dirs])

    autocast = torch.cuda.amp.autocast if amp else suppress
    feature_train_loader, feature_val_loader, feature_train_val_loader, feature_test_loader = get_combined_feature_dl(
        feature_dirs, batch_size, num_workers, fewshot_k, feature_combiner_cls, use_val_ds, normalize)

    if use_val_ds:
        best_wd = tune_weight_decay(feature_train_loader, feature_val_loader,
                         lr, epochs, autocast, device, verbose, seed)
        train_loader = feature_train_val_loader
    else:
        best_wd = 0
        train_loader = feature_train_loader

    return _evaluate(train_loader=train_loader,
                     best_wd=best_wd,
                     fewshot_k=fewshot_k,
                     feature_test_loader=feature_test_loader,
                     lr=lr,
                     epochs=epochs,
                     seed=seed,
                     device=device,
                     autocast=autocast,
                     normalize=normalize,
                     out_fn=out_fn,
                     verbose=verbose)
