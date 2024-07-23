import os
from enum import Enum
from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Regularization(Enum):
    L1 = 'L1'
    L2 = 'L2'
    weight_decay = 'weight_decay'


class Regularizer:
    def __init__(self, reg_type: str, reg_lambda: float):
        try:
            self.reg_type = Regularization(reg_type)
        except ValueError as e:
            raise ValueError(f"Regularization type {reg_type} not supported. Choose from {list(Regularization)}") from e
        self.reg_lambda = reg_lambda
        self.reg_func = self._get_regularizer(self.reg_type)

    @staticmethod
    def _get_regularizer(reg_type: Regularization):
        if reg_type == Regularization.L1:
            return lambda x: torch.sum(torch.abs(x))
        elif reg_type == Regularization.L2:
            return lambda x: torch.sum(x ** 2)
        elif reg_type == Regularization.weight_decay:
            return lambda x: 0.0

    def reg_loss(self, model):
        reg_loss = 0.0
        for name, param in model.named_parameters():
            if 'bias' not in name:
                reg_loss += self.reg_func(param)
        return self.reg_lambda * reg_loss

    def get_lambda(self):
        return self.reg_lambda if self.reg_type == Regularization.weight_decay else 0.0


class LinearProbe:
    def __init__(
            self,
            reg_lambda: float,
            lr: float,
            epochs: int,
            device: str,
            seed: int,
            logit_filter: Optional[torch.Tensor] = None,
            regularization: str = "weight_decay",
            verbose: bool = False
    ):
        self.reg = Regularizer(regularization, reg_lambda)
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.model = None
        self.logit_filter = logit_filter
        self.verbose = verbose

    @staticmethod
    def assign_learning_rate(param_group: dict, new_lr: float):
        param_group["lr"] = new_lr

    @staticmethod
    def _warmup_lr(base_lr: float, warmup_length: Union[float, int], step: int):
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

    def train(self, dataloader: DataLoader, filename: str = None) -> torch.nn.Module:
        # We reset the seed to ensure that the model is initialized with the same weights every time
        torch.manual_seed(self.seed)

        if filename is not None and os.path.exists(filename):
            if self.verbose:
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
            weight_decay=self.reg.get_lambda(),
        )
        criterion = torch.nn.CrossEntropyLoss()

        len_loader = len(dataloader)
        scheduler = self.cosine_lr(optimizer, self.lr, 0., self.epochs * len_loader)

        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                step = i + epoch * len_loader
                scheduler(step)

                optimizer.zero_grad()

                pred = self.model(x)
                loss = criterion(pred, y)
                loss += self.reg.reg_loss(self.model)

                loss.backward()
                optimizer.step()

            if self.verbose:
                print(
                    f"Train Epoch: {epoch} \t"
                    f"Loss: {loss.item():.6f}\t"
                    f"LR {optimizer.param_groups[0]['lr']:.5f}"
                )

        if filename is not None:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            if self.verbose:
                print(f"Saving model to {filename}")
            torch.save(self.model, filename)

        return self.model

    def infer(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        true, pred = [], []
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                if self.logit_filter is not None:
                    logits = logits @ self.logit_filter.T

                pred.append(logits.cpu())
                true.append(y.cpu())

        logits = torch.cat(pred)
        target = torch.cat(true)
        return logits, target
