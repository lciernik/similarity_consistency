import numpy as np

from clip_benchmark.eval.metrics import accuracy
from clip_benchmark.tasks.linear_probe import LinearProbe


class WeightDecayTuner:
    def __init__(self, lr, epochs, autocast, device, verbose, seed):
        self.lr = lr
        self.epochs = epochs
        self.autocast = autocast
        self.device = device
        self.verbose = verbose
        self.seed = seed

    def find_peak(self, wd_list, idxs, train_loader, val_loader):
        best_wd_idx, max_acc = 0, 0
        for idx in idxs:
            weight_decay = wd_list[idx]
            linear_probe = LinearProbe(weight_decay, self.lr, self.epochs, self.autocast, self.device, self.seed)
            linear_probe.train(train_loader)
            logits, target = linear_probe.infer(val_loader)
            acc1, = accuracy(logits.float(), target.float(), topk=(1,))
            if self.verbose:
                print(f"Valid accuracy with weight_decay {weight_decay}: {acc1}")
            if max_acc < acc1:
                best_wd_idx, max_acc = idx, acc1
        return best_wd_idx

    def tune_weight_decay(self, feature_train_loader, feature_val_loader):
        # perform openAI-like hyperparameter sweep
        # https://arxiv.org/pdf/2103.00020.pdf A.3
        # instead of scikit-learn LBFGS use FCNNs with AdamW
        wd_list = np.logspace(-6, 2, num=97).tolist()
        wd_list_init = np.logspace(-6, 2, num=7).tolist()
        wd_init_idx = [i for i, val in enumerate(wd_list) if val in wd_list_init]
        peak_idx = self.find_peak(wd_list, wd_init_idx, feature_train_loader, feature_val_loader)
        step_span = 8
        while step_span > 0:
            left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(wd_list) - 1)
            peak_idx = self.find_peak(wd_list, [left, peak_idx, right], feature_train_loader, feature_val_loader)
            step_span //= 2
        best_wd = wd_list[peak_idx]
        return best_wd