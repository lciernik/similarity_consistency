import torch


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
