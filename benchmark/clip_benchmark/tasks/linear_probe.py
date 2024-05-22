import os
import pickle
import time
from contextlib import suppress

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

from clip_benchmark.data.data_loader import get_feature_dl, get_combined_feature_dl
from clip_benchmark.data.data_utils import get_fewshot_indices
from clip_benchmark.eval.metrics import compute_metrics, accuracy
from clip_benchmark.models.featurizer import Featurizer


class LinearProbe:
    def __init__(self, weight_decay, lr, epochs, autocast, device, seed):
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.autocast = autocast
        self.device = device
        self.seed = seed
        self.model = None

    @staticmethod
    def assign_learning_rate(param_group, new_lr):
        param_group["lr"] = new_lr

    @staticmethod
    def _warmup_lr(base_lr, warmup_length, step):
        return base_lr * (step + 1) / warmup_length

    def cosine_lr(self, optimizer, base_lrs, warmup_length, steps):
        if not isinstance(base_lrs, list):
            base_lrs = [base_lrs for _ in optimizer.param_groups]
        assert len(base_lrs) == len(optimizer.param_groups)

        def _lr_adjuster(step):
            for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
                if step < warmup_length:
                    lr = self._warmup_lr(base_lr, warmup_length, step)
                else:
                    e = step - warmup_length
                    es = steps - warmup_length
                    lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
                self.assign_learning_rate(param_group, lr)

        return _lr_adjuster

    def train(self, dataloader, filename: str = None):
        torch.manual_seed(self.seed)

        if filename is not None and os.path.exists(filename):
            print(f"Loading model from {filename}")
            self.model = torch.load(filename)
            self.model = self.model.to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=[x for x in range(torch.cuda.device_count())])
            return

        input_shape, output_shape = dataloader.dataset[0][0].shape[0], dataloader.dataset.targets.max().item() + 1
        self.model = torch.nn.Linear(input_shape, output_shape)
        devices = [x for x in range(torch.cuda.device_count())]
        self.model = self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=devices)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss()

        len_loader = len(dataloader)
        scheduler = self.cosine_lr(optimizer, self.lr, 0., self.epochs * len_loader)

        for epoch in range(self.epochs):
            end = time.time()
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                step = i + epoch * len_loader
                data_time = time.time() - end
                scheduler(step)

                optimizer.zero_grad()
                with self.autocast():
                    pred = self.model(x)
                    loss = criterion(pred, y)

                loss.backward()
                optimizer.step()

                batch_time = time.time() - end
                end = time.time()

                if (i % 20) == 1:
                    num_samples = i * len(x)
                    try:
                        percent_complete = 100.0 * i / len_loader
                        progress_message = f"[{num_samples}/{len_loader} ({percent_complete:.0f}%)]"
                    except TypeError:
                        progress_message = f"[{num_samples} samples]"
                    print(
                        f"Train Epoch: {epoch} {progress_message}\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}\t"
                        f"LR {optimizer.param_groups[0]['lr']:.5f}"
                    )

        if filename is not None:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            print(f"Saving model to {filename}")
            torch.save(self.model, filename)

        return self.model

    def infer(self, dataloader):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        true, pred = [], []
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                with self.autocast():
                    logits = self.model(x)

                pred.append(logits.cpu())
                true.append(y.cpu())

        logits = torch.cat(pred)
        target = torch.cat(true)
        return logits, target


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


class BaseEvaluator:
    def __init__(self,
                 batch_size, num_workers, lr, epochs, seed, device,
                 fewshot_k, normalize=True, amp=True,
                 probe_out_dir=None, verbose=False, val_proportion=0):
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
        self.probe_out_dir = probe_out_dir
        self.verbose = verbose
        self.val_proportion = val_proportion

        self.wd_tuner = WeightDecayTuner(self.lr, self.epochs, self.autocast, self.device, self.verbose, self.seed)

        if probe_out_dir:
            self.probe_out_dir = self.check_single_instance(probe_out_dir, "pretrained model directory")
            self.linear_probe_fn = os.path.join(self.probe_out_dir, 'model.pkl')
        else:
            self.probe_out_dir = None
            self.linear_probe_fn = None

    @staticmethod
    def check_single_instance(param, param_name):
        if isinstance(param, list):
            if len(param) > 1:
                raise ValueError(f"SingleModelEvaluator only supports a single {param_name}.")
            return param[0]
        return param

    @staticmethod
    def check_feature_existence(feature_dir, verbose=False):
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
            stratify=targets
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

    def store_test_set_predictions(self, logits, target):
        if self.probe_out_dir:
            with open(os.path.join(self.probe_out_dir, 'predictions.pkl'), 'wb') as f:
                pickle.dump({'logits': logits, 'pred': logits.argmax(dim=1), 'target': target}, f)
                if self.verbose:
                    print(f"Stored test predictions in {os.path.join(self.probe_out_dir, 'test_predictions.pkl')}.")
        else:
            if self.verbose:
                print("No probe output directory specified. Not storing test set predictions.")

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
    def __init__(self,
                 batch_size, num_workers, lr, epochs, seed, device,
                 fewshot_k, model_id, feature_dir, model=None,
                 train_dataloader=None, eval_dataloader=None, normalize=True,
                 amp=True, probe_out_dir=None, verbose=False, val_proportion=0):
        super().__init__(batch_size, num_workers, lr, epochs, seed, device, fewshot_k, normalize, amp, probe_out_dir,
                         verbose, val_proportion)

        self.model_id = self.check_single_instance(model_id, "model id")
        self.feature_dir = self.check_single_instance(feature_dir, "feature directory")

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

        feature_train_loader, feature_test_loader = get_feature_dl(
            self.feature_dir, self.batch_size, self.num_workers, self.fewshot_k, self.use_val_ds)

        best_wd = self.optimize_weight_decay(feature_train_loader)

        return self._evaluate(train_loader=feature_train_loader,
                              test_loader=feature_test_loader,
                              best_wd=best_wd,
                              filename=self.linear_probe_fn)


class CombinedModelEvaluator(BaseEvaluator):
    def __init__(self, batch_size, num_workers, lr, epochs, seed, device,
                 fewshot_k, feature_dirs, feature_combiner_cls,
                 normalize=True, amp=True, probe_out_dir=None, verbose=False, val_proportion=0):

        super().__init__(batch_size, num_workers, lr, epochs, seed, device, fewshot_k, normalize, amp, probe_out_dir,
                         verbose, val_proportion)

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

    def evaluate(self):
        feature_train_loader, feature_test_loader = get_combined_feature_dl(
            self.feature_dirs, self.batch_size, self.num_workers, self.fewshot_k, self.feature_combiner_cls,
            self.normalize)

        best_wd = self.optimize_weight_decay(feature_train_loader, self.wd_tuner, self.val_proportion)

        return self._evaluate(train_loader=feature_train_loader,
                              test_loader=feature_test_loader,
                              best_wd=best_wd,
                              filename=self.linear_probe_fn)


class EnsembleModelEvaluator(BaseEvaluator):
    def __init__(self,
                 batch_size, num_workers, lr, epochs, seed, device,
                 fewshot_k, model_ids, feature_dirs, linear_prob_dirs,
                 normalize=True, amp=True, probe_out_dir=None, verbose=False, val_proportion=0):
        super().__init__(batch_size, num_workers, lr, epochs, seed, device, fewshot_k, normalize, amp, probe_out_dir,
                         verbose, val_proportion)
        self.model_ids = model_ids
        self.feature_dirs = feature_dirs
        self.linear_prob_dirs = linear_prob_dirs
        if not len(model_ids) == len(feature_dirs) == len(linear_prob_dirs):
            raise ValueError("Number of models, feature directories and linear probe directories must be the same.")

    def load_logits_targets(self, linear_probe_dir):
        with open(os.path.join(linear_probe_dir, 'predictions.pkl'), 'rb') as f:
            predictions = pickle.load(f)
            logits = predictions['logits']
            target = predictions['target']
            if self.verbose:
                print(f"Loaded test predictions from {os.path.join(linear_probe_dir, 'predictions.pkl')}.")
        return logits, target

    def retrain_linear_probe(self, idxs, feature_dir, model_fn):
        # TODO We do not store the metric of this retrained model (in comparison to single Evaluator)
        # Maybe one should remove the training and also rely on single Evaluator?
        feature_train_loader, feature_test_loader = get_feature_dl(
            feature_dir, self.batch_size, self.num_workers, self.fewshot_k, idxs)
        best_wd = self.optimize_weight_decay(feature_train_loader, self.wd_tuner, self.val_proportion)
        linear_probe = LinearProbe(weight_decay=best_wd,
                                   lr=self.lr,
                                   epochs=self.epochs,
                                   autocast=self.autocast,
                                   device=self.device,
                                   seed=self.seed)

        linear_probe.train(feature_train_loader, filename=model_fn)
        logits, target = linear_probe.infer(feature_test_loader)
        return logits, target

    def check_equal_targets(self, model_targets):
        if not all([torch.equal(model_targets[model_id], model_targets[self.model_ids[0]]) for model_id in
                    self.model_ids]):
            raise ValueError("Targets are not the same across models.")

    def ensemble_logits(self, model_logits, mode="post_softmax"):
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
        idxs = None
        model_logits = {}
        model_targets = {}
        for model_id, feature_dir, linear_probe_dir in zip(self.model_ids, self.feature_dirs, self.linear_prob_dirs):
            # Try to load predictions directly for maximum speed
            pred_fn = os.path.join(linear_probe_dir, 'predictions.pkl')
            if os.path.isfile(pred_fn):
                logits, target = self.load_logits_targets(model_id, linear_probe_dir)
                model_logits[model_id] = logits
                model_targets[model_id] = target
                continue

            model_fn = os.path.join(linear_probe_dir, 'model.pkl')

            # Retrain linear probe by loading precomputed features
            if idxs is None:
                targets = torch.load(os.path.join(feature_dir, 'targets_train.pt'))
                idxs = get_fewshot_indices(targets, self.fewshot_k)

            logits, target = self.retrain_linear_probe(idxs, feature_dir, model_fn)
            model_logits[model_id] = logits
            model_targets[model_id] = target

            # Save model Predictions
            with open(pred_fn, 'wb') as f:
                pickle.dump({'logits': logits, 'target': target}, f)
                if self.verbose:
                    print(f"Stored test predictions in {pred_fn}.")

        self.check_equal_targets(model_targets)

        logits = self.ensemble_logits(model_logits)
        self.store_test_set_predictions(logits, model_targets[self.model_ids[0]])
        metric_dict = compute_metrics(logits, model_targets[self.model_ids[0]])
        metric_dict = {"test_metrics": metric_dict, "weight_decay": best_wd}
        return metric_dict
