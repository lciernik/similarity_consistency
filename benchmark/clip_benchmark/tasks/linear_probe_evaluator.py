import os
import pickle
from typing import List, Union, Any, Optional, Tuple, Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from clip_benchmark.data.data_loader import get_feature_dl, get_combined_feature_dl
from clip_benchmark.data.data_utils import SubsetWithTargets
from clip_benchmark.data.feature_combiner import ConcatFeatureCombiner, PCAConcatFeatureCombiner
from clip_benchmark.eval.metrics import compute_metrics
from clip_benchmark.models.featurizer import Featurizer
from clip_benchmark.tasks.hyperparameter_tuner import HyperparameterTuner
from clip_benchmark.tasks.linear_probe import LinearProbe


class BaseEvaluator:
    def __init__(
            self, batch_size: int, num_workers: int, lrs: List[float], epochs: int, seed: int, device: str,
            fewshot_k: int, model_dirs: Optional[List[str]], predictions_dir: Optional[str],
            normalize: bool = True, verbose: bool = False, val_proportion: float = 0,
            logit_filter: Optional[torch.Tensor] = None, reg_lambda: float = 0.0, regularization: str = "weight_decay",
            force_train: bool = False
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.seed = seed
        self.device = device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, please run on a machine with a GPU.")

        self.normalize = normalize
        self.fewshot_k = fewshot_k
        if model_dirs is not None:
            self.linear_probe_fns = [os.path.join(model_dir, 'model.pkl') for model_dir in model_dirs]
        else:
            self.linear_probe_fns = None
        self.predictions_dir = predictions_dir
        self.verbose = verbose
        self.val_proportion = val_proportion
        self.logit_filter = logit_filter
        self.lrs = lrs
        self.reg_lambda = reg_lambda
        self.regularization = regularization
        self.force_train = force_train

        self.lr = None
        self.hp_tuner = HyperparameterTuner(
            lrs=lrs,
            epochs=self.epochs,
            regularization=self.regularization,
            device=self.device,
            verbose=self.verbose,
            seed=self.seed
        )

    @staticmethod
    def check_single_instance(param: List[Any], param_name: str) -> Union[Any, List[Any]]:
        if isinstance(param, list):
            if len(param) > 1:
                raise ValueError(f"Only supports a single {param_name} expected.")
            return param[0]
        return param

    @staticmethod
    def check_feature_existence(feature_dir: str, check_train: bool = True, verbose=False) -> bool:
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir, exist_ok=True)
            if verbose:
                print(f'Create path to store features: {feature_dir}')
            return False
        filenames_to_check = ['features_test.pt', 'targets_test.pt']
        if check_train:
            filenames_to_check += ['features_train.pt', 'targets_train.pt']
        all_exist = True
        for filename in filenames_to_check:
            if not os.path.exists(os.path.join(feature_dir, filename)):
                all_exist = False
                if verbose:
                    print(f"File {filename} is missing in {feature_dir}.")
                break
        return all_exist

    def _create_train_val_loaders(self, train_loader: DataLoader) -> Tuple[DataLoader, DataLoader]:
        train_dataset = train_loader.dataset
        targets = np.array(train_dataset.targets)
        train_indices, val_indices = train_test_split(
            np.arange(targets.shape[0]),
            test_size=self.val_proportion,
            stratify=targets,
            random_state=self.seed
        )
        tmp_train_dataset = SubsetWithTargets(train_dataset, indices=train_indices)
        tmp_val_dataset = SubsetWithTargets(train_dataset, indices=val_indices)

        tmp_train_loader = DataLoader(tmp_train_dataset, batch_size=train_loader.batch_size,
                                      shuffle=True, num_workers=train_loader.num_workers)
        tmp_val_loader = DataLoader(tmp_val_dataset, batch_size=train_loader.batch_size,
                                    shuffle=True, num_workers=train_loader.num_workers)
        return tmp_train_loader, tmp_val_loader

    def optimize_hyperparams(self, train_loader: DataLoader) -> None:
        if self.val_proportion > 0:
            if self.verbose:
                print(f"\nTuning hyperparameters of linear probe.\n")
            # Split train set into train and validation
            tmp_train_loader, tmp_val_loader = self._create_train_val_loaders(train_loader)
            best_lr, best_wd = self.hp_tuner.tune(tmp_train_loader, tmp_val_loader)
        else:
            if len(self.lrs) != 1:
                raise ValueError("Only one learning rate is supported without a validation set.")
            best_lr = self.lrs[0]
            best_wd = self.reg_lambda

        self.lr = best_lr
        self.reg_lambda = best_wd

    def load_test_set_predictions(self, linear_probe_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
        with open(os.path.join(linear_probe_dir, 'predictions.pkl'), 'rb') as f:
            predictions = pickle.load(f)
            logits = predictions['logits']
            target = predictions['target']
            if self.verbose:
                print(f"Loaded test predictions from {os.path.join(linear_probe_dir, 'predictions.pkl')}.")
        return logits, target

    def store_test_set_predictions(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if not os.path.exists(self.predictions_dir):
            os.makedirs(self.predictions_dir, exist_ok=True)
            if self.verbose:
                print(f'Create path to store predictions: {self.predictions_dir}')
        with open(os.path.join(self.predictions_dir, 'predictions.pkl'), 'wb') as f:
            pickle.dump({'logits': logits, 'pred': logits.argmax(dim=1), 'target': target}, f)
            if self.verbose:
                print(f"Stored test predictions in {os.path.join(self.predictions_dir, 'predictions.pkl')}.")

    def _evaluate(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            filename: Optional[str] = None,
            evaluate_train: bool = True
    ) -> dict:
        linear_probe = LinearProbe(
            reg_lambda=self.reg_lambda,
            lr=self.lr,
            epochs=self.epochs,
            device=self.device,
            seed=self.seed,
            logit_filter=self.logit_filter,
            regularization=self.regularization,
            verbose=self.verbose
        )
        metric_dict = {
            "reg_lambda": self.reg_lambda,
            "learning_rate": self.lr,
        }

        linear_probe.train(train_loader, filename=filename, force_train=self.force_train)

        if evaluate_train:
            train_logits, train_targets = linear_probe.infer(train_loader)
            train_metrics = compute_metrics(train_logits, train_targets, self.verbose)
            metric_dict["train_metrics"] = train_metrics

        test_logits, test_targets = linear_probe.infer(test_loader)
        test_metrics = compute_metrics(test_logits, test_targets, self.verbose)
        metric_dict["test_metrics"] = test_metrics

        self.store_test_set_predictions(test_logits, test_targets)

        return metric_dict

    def evaluate(self) -> dict:
        raise NotImplementedError("Subclasses must implement this method")


class SingleModelEvaluator(BaseEvaluator):

    def __init__(
            self, batch_size: int, num_workers: int, lrs: List[float], epochs: int, seed: int, device: str,
            fewshot_k: int, feature_dirs: Optional[List[str]], model_dirs: Optional[List[str]],
            predictions_dir: Optional[str], model: Optional[Any] = None,
            train_dataloader: Optional[torch.utils.data.DataLoader] = None,
            eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
            normalize: bool = True, verbose: bool = False, val_proportion: float = 0,
            logit_filter: Optional[torch.Tensor] = None, reg_lambda: float = 0.0, regularization: str = "weight_decay",
            force_train: bool = False
    ) -> None:
        super().__init__(batch_size, num_workers, lrs, epochs, seed, device, fewshot_k, model_dirs, predictions_dir,
                         normalize, verbose, val_proportion, logit_filter, reg_lambda, regularization, force_train)

        self.feature_dir = self.check_single_instance(feature_dirs, "feature directory")
        self.linear_probe_fn = self.check_single_instance(self.linear_probe_fns, "linear probe filename")

        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

    def _check_model_and_loaders_existence(self) -> bool:
        # Check if model and dataloaders are provided.
        # If a model is provided at least one dataloader should be provided.
        return (self.model is not None) and ((self.train_dataloader is not None) or (self.eval_dataloader is not None))

    def ensure_feature_availability(self, check_train: bool = True):
        model_available = self._check_model_and_loaders_existence()
        check_train = check_train and self.train_dataloader is not None
        feat_available = self.check_feature_existence(self.feature_dir, check_train=check_train, verbose=self.verbose)
        if (not feat_available) and model_available:
            # We need to generate features if these do not exist
            featurizer = Featurizer(model=self.model, normalize=self.normalize).to(self.device)
            featurizer.feature_extraction(train_dataloader=self.train_dataloader,
                                          eval_dataloader=self.eval_dataloader,
                                          feature_dir=self.feature_dir,
                                          device=self.device)
        elif (not feat_available) and (not model_available):
            raise ValueError(f"Features are missing (in {self.feature_dir})\nand no model and dataloaders are provided."
                             f"Please run feature extraction first.")
        else:
            if self.verbose:
                print(f"Features are already available in {self.feature_dir}.")

    def evaluate(self) -> dict:
        probe_exists = os.path.exists(self.linear_probe_fn) and not self.force_train
        
        if self.verbose:
            if probe_exists:
                print("Found probe at ", self.linear_probe_fn)
            else:
                print("Could not find probe at ", self.linear_probe_fn)

        self.ensure_feature_availability(check_train=not probe_exists)

        feature_train_loader, feature_test_loader = get_feature_dl(feature_dir=self.feature_dir,
                                                                   batch_size=self.batch_size,
                                                                   num_workers=self.num_workers,
                                                                   fewshot_k=self.fewshot_k,
                                                                   load_train=not probe_exists)
        if probe_exists:
            self.reg_lambda = None
        else:
            self.optimize_hyperparams(feature_train_loader)

        return self._evaluate(train_loader=feature_train_loader,
                              test_loader=feature_test_loader,
                              filename=self.linear_probe_fn ,
                              evaluate_train=not probe_exists)


class CombinedModelEvaluator(BaseEvaluator):
    def __init__(
            self, batch_size: int, num_workers: int, lrs: List[float], epochs: int, seed: int, device: str,
            fewshot_k: int, feature_dirs: Optional[List[str]], model_dirs: Optional[List[str]],
            predictions_dir: Optional[str],
            feature_combiner_cls: Union[ConcatFeatureCombiner, PCAConcatFeatureCombiner],
            normalize: bool = True, verbose: bool = False, val_proportion: float = 0,
            logit_filter: Optional[torch.Tensor] = None, reg_lambda: float = 0.0, regularization: str = "weight_decay",
            force_train: bool = False
    ) -> None:

        super().__init__(batch_size, num_workers, lrs, epochs, seed, device, fewshot_k, model_dirs, predictions_dir,
                         normalize, verbose, val_proportion, logit_filter, reg_lambda, regularization, force_train)
        self.feature_dirs = feature_dirs
        self.feature_combiner_cls = feature_combiner_cls
        self.linear_probe_fn = self.check_single_instance(self.linear_probe_fns, "linear probe filename")

    def require_feature_existence(self, check_train: bool = True):
        available_features = [self.check_feature_existence(feature_dir, check_train=check_train, verbose=self.verbose)
                              for feature_dir in
                              self.feature_dirs]
        if not all(available_features):
            not_available_features = [feature_dir for feature_dir, available in
                                      zip(self.feature_dirs, available_features) if
                                      not available]
            raise ValueError(f"Features are missing in {not_available_features}, please run single evaluator first!")

    def evaluate(self) -> dict:
        probe_exists = os.path.exists(self.linear_probe_fn) and not self.force_train

        self.require_feature_existence(check_train=not probe_exists)

        feature_train_loader, feature_test_loader = get_combined_feature_dl(feature_dirs=self.feature_dirs,
                                                                            batch_size=self.batch_size,
                                                                            num_workers=self.num_workers,
                                                                            fewshot_k=self.fewshot_k,
                                                                            feature_combiner_cls=self.feature_combiner_cls,
                                                                            normalize=self.normalize,
                                                                            load_train=not probe_exists)

        if probe_exists:
            self.reg_lambda = None
        else:
            self.optimize_hyperparams(feature_train_loader)

        return self._evaluate(train_loader=feature_train_loader,
                              test_loader=feature_test_loader,
                              filename=self.linear_probe_fn,
                              evaluate_train=not probe_exists)


class EnsembleModelEvaluator(BaseEvaluator):
    def __init__(
            self, batch_size: int, num_workers: int, lrs: List[float], epochs: int, seed: int, device: str,
            fewshot_k: int, model_ids: List[str], feature_dirs: Optional[List[str]], model_dirs: Optional[List[str]],
            predictions_dir: Optional[str], single_prediction_dirs: Optional[List[str]], normalize: bool = True,
            verbose: bool = False, val_proportion: float = 0, logit_filter: Optional[torch.Tensor] = None,
            reg_lambda: float = 0.0, regularization: str = "weight_decay", force_train: bool = False
    ) -> None:
        super().__init__(batch_size, num_workers, lrs, epochs, seed, device, fewshot_k, model_dirs, predictions_dir,
                         normalize, verbose, val_proportion, logit_filter, reg_lambda, regularization, False)
        self.model_ids = model_ids
        self.feature_dirs = feature_dirs
        self.single_prediction_dirs = single_prediction_dirs
        if not len(model_ids) == len(feature_dirs) == len(single_prediction_dirs):
            raise ValueError("Number of models, feature, single model prediction, and  linear probe directories "
                             "must be the same.")

    def check_equal_targets(self, model_targets: Dict[str, torch.Tensor]):
        if not all([torch.equal(model_targets[model_id], model_targets[self.model_ids[0]]) for model_id in
                    self.model_ids]):
            raise ValueError("Targets are not the same across models.")

    @staticmethod
    def ensemble_logits(model_logits: Dict[str, torch.Tensor], mode: str = "post_softmax") -> torch.Tensor:
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

    def evaluate(self) -> dict:
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
