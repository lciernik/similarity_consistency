import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class Featurizer(torch.nn.Module):
    def __init__(self, model, normalize=True):
        super().__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, input):
        image_features = self.model.encode_image(input)
        if self.normalize:
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def feature_extraction(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        feature_dir: str,
        device: str,
    ):
        """
        Extract features from the dataset using the featurizer model and store them in the feature_dir
        """
        splits = ["_train", "_test"]
        for save_str, loader in zip(splits, [train_dataloader, eval_dataloader]):
            if loader is None:
                continue
            features = []
            targets = []
            num_batches_tracked = 0
            num_cached = 0
            with torch.no_grad():
                for images, target in tqdm(loader):
                    images = images.to(device)

                    feature = self.forward(images)

                    features.append(feature.cpu())
                    targets.append(target)

                    num_batches_tracked += 1
                    if (num_batches_tracked % 100) == 0:
                        features = torch.cat(features)
                        targets = torch.cat(targets)

                        torch.save(
                            features,
                            os.path.join(
                                feature_dir, f"features{save_str}_cache_{num_cached}.pt"
                            ),
                        )
                        torch.save(
                            targets,
                            os.path.join(
                                feature_dir, f"targets{save_str}_cache_{num_cached}.pt"
                            ),
                        )
                        num_cached += 1
                        features = []
                        targets = []

            if len(features) > 0:
                features = torch.cat(features)
                targets = torch.cat(targets)
                torch.save(
                    features,
                    os.path.join(
                        feature_dir, f"features{save_str}_cache_{num_cached}.pt"
                    ),
                )
                torch.save(
                    targets,
                    os.path.join(
                        feature_dir, f"targets{save_str}_cache_{num_cached}.pt"
                    ),
                )
                num_cached += 1

            features = torch.load(
                os.path.join(feature_dir, f"features{save_str}_cache_0.pt")
            )
            targets = torch.load(
                os.path.join(feature_dir, f"targets{save_str}_cache_0.pt")
            )
            for k in range(1, num_cached):
                next_features = torch.load(
                    os.path.join(feature_dir, f"features{save_str}_cache_{k}.pt")
                )
                next_targets = torch.load(
                    os.path.join(feature_dir, f"targets{save_str}_cache_{k}.pt")
                )
                features = torch.cat((features, next_features))
                targets = torch.cat((targets, next_targets))

            for k in range(num_cached):
                os.remove(os.path.join(feature_dir, f"features{save_str}_cache_{k}.pt"))
                os.remove(os.path.join(feature_dir, f"targets{save_str}_cache_{k}.pt"))

            torch.save(features, os.path.join(feature_dir, f"features{save_str}.pt"))
            torch.save(targets, os.path.join(feature_dir, f"targets{save_str}.pt"))
