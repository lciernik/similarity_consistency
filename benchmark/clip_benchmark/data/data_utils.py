import numpy as np
import torch
from torch.utils.data import Subset

from clip_benchmark.data import build_dataset, get_dataset_collate_fn
from clip_benchmark.models import load_model


def get_fewshot_indices(targets, fewshot_k):
    """
    Get the indices of the features that are use for training the linear probe
    """
    length = len(targets)
    perm = [p.item() for p in torch.randperm(length)]
    idxs = []
    counts = {}
    num_classes = 0

    for p in perm:
        target = targets[p].item()
        if target not in counts:
            counts[target] = 0
            num_classes += 1

        if fewshot_k < 0 or counts[target] < fewshot_k:
            counts[target] += 1
            idxs.append(p)

    for c in counts:
        if fewshot_k > 0 and counts[c] != fewshot_k:
            raise ValueError(f'insufficient data for eval with {fewshot_k} samples per class, '
                             f'only {counts[c]} samples for class {c}')

    return idxs


def get_extraction_model_n_dataloader(args, dataset_root, task):
    assert isinstance(args.model, str), "model should be a string"
    if args.verbose:
        print(
            f"Load model and use {'no' if args.feature_alignment is None else args.feature_alignment} feature "
            f"alignment",
            flush=True)
    model, transform = load_model(
        model_name=args.model,
        source=args.model_source,
        model_parameters=args.model_parameters,
        module_name=args.module_name,
        feature_alignment=args.feature_alignment,
        device=args.device
    )

    eval_dataset = build_dataset(
        dataset_name=args.dataset,
        root=dataset_root,
        transform=transform,
        split=args.split,  # by default this is the test split
        download=True,
        wds_cache_dir=args.wds_cache_dir,
        verbose=args.verbose
    )
    collate_fn = get_dataset_collate_fn(args.dataset)
    if args.verbose:
        try:
            print(f"Dataset size: {len(eval_dataset)}")
        except TypeError:
            print("IterableDataset has no len()")
        print(f"Dataset split: {args.split}")
        if hasattr(eval_dataset, "classes") and eval_dataset.classes:
            try:
                print(f"Dataset classes: {eval_dataset.classes}")
                print(f"Dataset number of classes: {len(eval_dataset.classes)}")
            except AttributeError:
                print("Dataset has no classes.")

    # Get the dataloader for the split we want to evaluate on, by default this is the test split
    if args.dataset.startswith("wds/"):
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset.batched(args.batch_size), batch_size=None,
            shuffle=False, num_workers=args.num_workers,
        )
    else:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_fn
        )
    # we also need the train and validation splits for linear probing.
    train_dataset = build_dataset(
        dataset_name=args.dataset,
        root=dataset_root,
        transform=transform,
        split=args.train_split,
        download=True,
        verbose=args.verbose
    )
    if train_dataset:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            collate_fn=collate_fn, pin_memory=True,
        )
    else:
        train_dataloader = None

    return model, train_dataloader, eval_dataloader


class SubsetWithTargets(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = np.array([dataset.targets[i] for i in indices])
