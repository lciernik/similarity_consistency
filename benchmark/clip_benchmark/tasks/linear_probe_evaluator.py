import os
import pickle
from contextlib import suppress
from typing import List, Union, Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from clip_benchmark.data.data_loader import get_feature_dl, get_combined_feature_dl
from clip_benchmark.eval.metrics import compute_metrics
from clip_benchmark.models.featurizer import Featurizer
from clip_benchmark.tasks.linear_probe import LinearProbe
from clip_benchmark.tasks.weight_decay_tuner import WeightDecayTuner


class BaseEvaluator:
    def __init__(self, batch_size: int, num_workers: int, lr: float, epochs: int, seed: int, device: str,
                 fewshot_k: int, model_dirs: List[str], predictions_dir: str, normalize: bool = True, amp: bool = True,
                 verbose: bool = False, val_proportion: float = 0) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.device = device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, please run on a machine with a GPU.")
        self.autocast = torch.cuda.amp.autocast if amp else suppress

        self.normalize = normalize
        self.fewshot_k = fewshot_k
        self.linear_probe_fns = [os.path.join(model_dir, 'model.pkl') for model_dir in model_dirs]
        self.predictions_dir = predictions_dir
        self.verbose = verbose
        self.val_proportion = val_proportion

        self.wd_tuner = WeightDecayTuner(self.lr, self.epochs, self.autocast, self.device, self.verbose, self.seed)

    @staticmethod
    def check_single_instance(param: List[Any], param_name: str) -> Union[Any, List[Any]]:
        if isinstance(param, list):
            if len(param) > 1:
                raise ValueError(f"Only supports a single {param_name} expected.")
            return param[0]
        return param

    @staticmethod
    def check_feature_existence(feature_dir: str, verbose=False) -> bool:
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir, exist_ok=True)
            if verbose:
                print(f'Create path to store features: {feature_dir}')
            return False
        if not os.path.exists(os.path.join(feature_dir, 'targets_train.pt')):
            return False
        return True

    def _create_train_val_loaders(self, train_loader):
        train_dataset = train_loader.dataset
        targets = np.array(train_dataset.targets)
        train_indices, val_indices = train_test_split(
            np.arange(targets.shape[0]),
            test_size=self.val_proportion,
            stratify=targets,
            random_state=self.seed
        )
        tmp_train_dataset = Subset(train_dataset, indices=train_indices)
        tmp_val_dataset = Subset(train_dataset, indices=val_indices)

        tmp_train_loader = DataLoader(tmp_train_dataset, batch_size=train_loader.batch_size,
                                      shuffle=True, num_workers=train_loader.num_workers)
        tmp_val_loader = DataLoader(tmp_val_dataset, batch_size=train_loader.batch_size,
                                    shuffle=True, num_workers=train_loader.num_workers)
        return tmp_train_loader, tmp_val_loader

    def optimize_weight_decay(self, train_loader):
        if self.val_proportion > 0:
            # Split train set into train and validation
            tmp_train_loader, tmp_val_loader = self._create_train_val_loaders(train_loader)
            best_wd = self.wd_tuner.tune_weight_decay(tmp_train_loader, tmp_val_loader)
        else:
            # TODO Enable Weight Decay Settings without Validation Set
            best_wd = 0

        return best_wd

    def load_test_set_predictions(self, linear_probe_dir):
        with open(os.path.join(linear_probe_dir, 'predictions.pkl'), 'rb') as f:
            predictions = pickle.load(f)
            logits = predictions['logits']
            target = predictions['target']
            if self.verbose:
                print(f"Loaded test predictions from {os.path.join(linear_probe_dir, 'predictions.pkl')}.")
        return logits, target

    def store_test_set_predictions(self, logits, target):
        if not os.path.exists(self.predictions_dir):
            os.makedirs(self.predictions_dir, exist_ok=True)
            if self.verbose:
                print(f'Create path to store predictions: {self.predictions_dir}')
        with open(os.path.join(self.predictions_dir, 'predictions.pkl'), 'wb') as f:
            pickle.dump({'logits': logits, 'pred': logits.argmax(dim=1), 'target': target}, f)
            if self.verbose:
                print(f"Stored test predictions in {os.path.join(self.predictions_dir, 'predictions.pkl')}.")

    def _evaluate(self, train_loader, test_loader, best_wd, filename=None):
        linear_probe = LinearProbe(weight_decay=best_wd,
                                   lr=self.lr,
                                   epochs=self.epochs,
                                   autocast=self.autocast,
                                   device=self.device,
                                   seed=self.seed)
        linear_probe.train(train_loader, filename=filename)

        train_logits, train_targets = linear_probe.infer(train_loader)
        train_metrics = compute_metrics(train_logits, train_targets, self.verbose)

        test_logits, test_targets = linear_probe.infer(test_loader)
        test_metrics = compute_metrics(test_logits, test_targets, self.verbose)

        self.store_test_set_predictions(test_logits, test_targets)

        metric_dict = {"train_metrics": train_metrics, "test_metrics": test_metrics, "best_weight_decay": best_wd}

        return metric_dict

    def evaluate(self):
        raise NotImplementedError("Subclasses must implement this method")


class SingleModelEvaluator(BaseEvaluator):
    def __init__(self, batch_size, num_workers, lr, epochs, seed, device, fewshot_k, feature_dirs, model_dirs,
                 predictions_dir, model=None, train_dataloader=None, eval_dataloader=None, normalize=True,
                 amp=True, verbose=False, val_proportion=0):
        super().__init__(batch_size, num_workers, lr, epochs, seed, device, fewshot_k, model_dirs, predictions_dir,
                         normalize, amp, verbose, val_proportion)

        self.feature_dir = self.check_single_instance(feature_dirs, "feature directory")
        self.linear_probe_fn = self.check_single_instance(self.linear_probe_fns, "linear probe filename")

        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

    def ensure_feature_availability(self):
        if not self.check_feature_existence(self.feature_dir, self.verbose):
            # We need to generate features if these do not exist
            featurizer = Featurizer(model=self.model, normalize=self.normalize).to(self.device)
            featurizer.feature_extraction(train_dataloader=self.train_dataloader,
                                          eval_dataloader=self.eval_dataloader,
                                          feature_dir=self.feature_dir,
                                          device=self.device,
                                          autocast=self.autocast)

    def evaluate(self):
        self.ensure_feature_availability()

        feature_train_loader, feature_test_loader = get_feature_dl(feature_dir=self.feature_dir,
                                                                   batch_size=self.batch_size,
                                                                   num_workers=self.num_workers,
                                                                   fewshot_k=self.fewshot_k)

        best_wd = self.optimize_weight_decay(feature_train_loader)

        return self._evaluate(train_loader=feature_train_loader,
                              test_loader=feature_test_loader,
                              best_wd=best_wd,
                              filename=self.linear_probe_fn)


class CombinedModelEvaluator(BaseEvaluator):
    def __init__(self, batch_size, num_workers, lr, epochs, seed, device, fewshot_k, feature_dirs, model_dirs,
                 predictions_dir, feature_combiner_cls, normalize=True, amp=True, verbose=False, val_proportion=0):

        super().__init__(batch_size, num_workers, lr, epochs, seed, device, fewshot_k, model_dirs, predictions_dir,
                         normalize, amp, verbose, val_proportion)

        # TODO: maybe this is bad to do it here, as the storing paths contain all the models
        available_features = [self.check_feature_existence(feature_dir, verbose) for feature_dir in feature_dirs]
        if not any(available_features):
            raise ValueError("Features of all models are missing. Please generate features first.")
        if not all(available_features):
            self.feature_dirs = [feature_dir for feature_dir, available in zip(feature_dirs, available_features) if
                                 available]
            if self.verbose:
                print(f"Using only available features: {self.feature_dirs}")
        else:
            self.feature_dirs = feature_dirs

        self.feature_combiner_cls = feature_combiner_cls
        self.linear_probe_fn = self.check_single_instance(self.linear_probe_fns, "linear probe filename")

    def evaluate(self):
        feature_train_loader, feature_test_loader = get_combined_feature_dl(feature_dirs=self.feature_dirs,
                                                                            batch_size=self.batch_size,
                                                                            num_workers=self.num_workers,
                                                                            fewshot_k=self.fewshot_k,
                                                                            feature_combiner_cls=self.feature_combiner_cls,
                                                                            normalize=self.normalize)

        best_wd = self.optimize_weight_decay(feature_train_loader)

        return self._evaluate(train_loader=feature_train_loader,
                              test_loader=feature_test_loader,
                              best_wd=best_wd,
                              filename=self.linear_probe_fn)


class EnsembleModelEvaluator(BaseEvaluator):
    def __init__(self, batch_size, num_workers, lr, epochs, seed, device, fewshot_k, model_ids,
                 feature_dirs, model_dirs, predictions_dir, single_prediction_dirs,
                 normalize=True, amp=True, verbose=False, val_proportion=0):

        super().__init__(batch_size, num_workers, lr, epochs, seed, device, fewshot_k, model_dirs, predictions_dir,
                         normalize, amp, verbose, val_proportion)

        self.model_ids = model_ids
        self.feature_dirs = feature_dirs
        self.single_prediction_dirs = single_prediction_dirs
        if not len(model_ids) == len(feature_dirs) == len(single_prediction_dirs):
            raise ValueError("Number of models, feature, single model prediction, and  linear probe directories "
                             "must be the same.")

    def check_equal_targets(self, model_targets):
        if not all([torch.equal(model_targets[model_id], model_targets[self.model_ids[0]]) for model_id in
                    self.model_ids]):
            raise ValueError("Targets are not the same across models.")

    @staticmethod
    def ensemble_logits(model_logits, mode="post_softmax"):
        if mode == "post_softmax":
            # Softmax does not work for float16
            probs = torch.stack(
                [torch.nn.functional.softmax(logits.float(), dim=1) for logits in model_logits.values()],
                dim=0)
            logits = torch.mean(probs, dim=0)
        elif mode == "pre_softmax":
            logits = torch.mean(torch.stack([logits for logits in model_logits.values()], dim=0), dim=1)
        else:
            raise ValueError(f"Unknown mode {mode}")
        return logits

    def evaluate(self):
        model_logits = {}
        model_targets = {}
        for model_id, feature_dir, model_pred_dir in zip(self.model_ids, self.feature_dirs,
                                                         self.single_prediction_dirs):
            # Try to load predictions directly for maximum speed
            pred_fn = os.path.join(model_pred_dir, 'predictions.pkl')
            if os.path.isfile(pred_fn):
                logits, target = self.load_test_set_predictions(model_pred_dir)
                model_logits[model_id] = logits
                model_targets[model_id] = target
            else:
                raise ValueError(
                    f"Predictions for model {model_id} are missing, please run single evaluator for {model_id} first!"
                )

        self.check_equal_targets(model_targets)

        logits = self.ensemble_logits(model_logits)
        self.store_test_set_predictions(logits, model_targets[self.model_ids[0]])
        metric_dict = compute_metrics(logits, model_targets[self.model_ids[0]])
        metric_dict = {"test_metrics": metric_dict}
        return metric_dict