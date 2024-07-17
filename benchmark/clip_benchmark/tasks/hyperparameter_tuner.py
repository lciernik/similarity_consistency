from typing import List, Tuple

from torch.utils.data import DataLoader

from clip_benchmark.tasks.weight_decay_tuner import WeightDecayTuner


class HyperparameterTuner:
    """
    This class is used to find the best hyperparameter setting (learning rate and weight decay) for the linear probe.
    It uses the WeightDecay tuner to find the best weight decay for each combination of the other hyperparameters.
    """

    def __init__(self, lrs: List[float], epochs: int, device: str, verbose: bool, seed: int, weight_decay_type:str="L2"):
        self.lrs = lrs
        self.weight_decay_type = weight_decay_type
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.seed = seed

    def tune(self, feature_train_loader: DataLoader, feature_val_loader:DataLoader) -> Tuple[float, float]:
        best_lr, best_wd, max_acc = 0, 0, 0
        for lr in self.lrs:
            weight_decay_tuner = WeightDecayTuner(lr, self.epochs, self.device, self.verbose, self.seed)
            weight_decay, acc1 = weight_decay_tuner.tune_weight_decay(feature_train_loader, feature_val_loader)
            if self.verbose:
                print(f"\nValid accuracy with lr {lr} and weight_decay {weight_decay}: {acc1}\n")
            if max_acc < acc1:
                best_lr, best_wd, max_acc = lr, weight_decay, acc1
        return best_lr, best_wd
