import os
import time

import numpy as np
import torch
from tqdm import tqdm


class LinearProbe:
    def __init__(self, weight_decay, lr, epochs, autocast, device, seed, logit_filter=None):
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

    def train(self, dataloader, filename=None):
        # We reset the seed to ensure that the model is initialized with the same weights every time
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
                    if self.logit_filter is not None:
                        logits = logits.T[self.logit_filter].T

                pred.append(logits.cpu())
                true.append(y.cpu())

        logits = torch.cat(pred)
        target = torch.cat(true)
        return logits, target
