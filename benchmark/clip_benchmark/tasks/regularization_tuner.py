from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader

from clip_benchmark.eval.metrics import accuracy
from clip_benchmark.tasks.linear_probe import LinearProbe


class RegularizationTuner:
    def __init__(
            self,
            lr: float,
            epochs: int,
            device: str,
            verbose: bool,
            seed: int,
            regularization: str = "weight_decay"
    ):
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.seed = seed
        self.regularization = regularization

    def find_peak(
            self,
            lambda_list: List[float],
            idxs: List[int],
            train_loader: DataLoader,
            val_loader: DataLoader
    ) -> Tuple[int, float]:
        best_lambda_idx, max_acc = 0, 0
        for idx in idxs:
            reg_lambda = lambda_list[idx]
            linear_probe = LinearProbe(
                reg_lambda=reg_lambda,
                lr=self.lr,
                epochs=self.epochs,
                device=self.device,
                seed=self.seed,
                regularization=self.regularization,
                verbose=self.verbose
            )
            linear_probe.train(train_loader)
            logits, target = linear_probe.infer(val_loader)
            acc1, = accuracy(logits.float(), target.float(), topk=(1,))
            if self.verbose:
                print(f"\nValid accuracy with regularization lambda {reg_lambda}: {acc1}\n")
            if max_acc <= acc1:
                best_lambda_idx, max_acc = idx, acc1
        return best_lambda_idx, max_acc

    def tune_lambda(
            self,
            feature_train_loader: DataLoader,
            feature_val_loader: DataLoader,
            min_exp: int = -6,
            max_exp: int = 1,
    ) -> Tuple[float, float]:
        # perform openAI-like hyperparameter sweep
        # https://arxiv.org/pdf/2103.00020.pdf A.3
        # instead of scikit-learn LBFGS use FCNNs with AdamW
        # lambda_list = np.logspace(-6, 2, num=97).tolist()
        # lambda_list_init = np.logspace(-6, 2, num=7).tolist()

        num_init = max_exp - min_exp + 1
        lambda_list_init = np.logspace(min_exp, max_exp, num=num_init).tolist()
        # Put 8 values between each lambda_list_init value
        lambda_list = np.logspace(min_exp, max_exp, num=(num_init + (num_init - 1) * 8)).tolist()
        lambda_init_idx = [lambda_list.index(val) for val in lambda_list_init]

        peak_idx, acc1 = self.find_peak(lambda_list, lambda_init_idx, feature_train_loader, feature_val_loader)
        # step_span = 8
        step_span = 4
        while step_span > 0:
            left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(lambda_list) - 1)
            # avoid testing the peak_idx
            idxs_to_test = [idx for idx in [left, right] if idx != peak_idx]
            if len(idxs_to_test) == 0:
                break
            new_peak_idx, new_acc1 = self.find_peak(lambda_list, idxs_to_test, feature_train_loader, feature_val_loader)
            if new_acc1 > acc1:
                acc1 = new_acc1
                peak_idx = new_peak_idx
            step_span //= 2
        best_lambda = lambda_list[peak_idx]
        return best_lambda, acc1
