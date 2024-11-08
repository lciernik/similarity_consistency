from typing import List, Tuple

from torch.utils.data import DataLoader

from sim_consistency.tasks.regularization_tuner import RegularizationTuner


class HyperparameterTuner:
    """
    This class is used to find the best hyperparameter setting (learning rate and regularizaation parameter) for the
    linear probe.  It uses the RegularizationTuner to find the best regularization value for each combination of the
    other hyperparameters.
    """

    def __init__(
            self,
            lrs: List[float],
            epochs: int,
            device: str,
            verbose: bool,
            seed: int,
            regularization: str = "weight_decay"
    ):
        self.lrs = lrs
        self.regularization = regularization
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.seed = seed

    def tune(self, feature_train_loader: DataLoader, feature_val_loader: DataLoader) -> Tuple[float, float]:
        best_lr, best_reg_lambda, max_acc = 0, 0, 0
        for lr in self.lrs:
            regularization_tuner = RegularizationTuner(lr, self.epochs, self.device, self.verbose, self.seed,
                                                       regularization=self.regularization)
            reg_lambda, acc1 = regularization_tuner.tune_lambda(feature_train_loader, feature_val_loader)
            if self.verbose:
                print(f"\nValid accuracy with lr {lr} and reg_lambda {reg_lambda}: {acc1}\n")
            if max_acc < acc1:
                best_lr, best_reg_lambda, max_acc = lr, reg_lambda, acc1
        return best_lr, best_reg_lambda
