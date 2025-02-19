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
        use_cache: bool = True,
        num_batches_in_cache: int = 100,
    ):
        """
        Extract features from the dataset using the featurizer model and store them in the feature_dir
        """
        splits = ["_train", "_test"]
        for save_str, loader in zip(splits, [train_dataloader, eval_dataloader]):
            if loader is None:
                continue

            num_cached = 0
            if use_cache:
                while os.path.exists(
                    os.path.join(
                        feature_dir, f"features{save_str}_cache_{num_cached}.pt"
                    )
                ) and os.path.exists(
                    os.path.join(
                        feature_dir, f"targets{save_str}_cache_{num_cached}.pt"
                    )
                ):
                    num_cached += 1
                if num_cached > 0:
                    print(
                        f"\nFound {num_cached} cached feature{save_str} and target{save_str} files in {feature_dir=}.\n"
                    )

            features = []
            targets = []
            num_batches_tracked = num_cached * num_batches_in_cache
            with torch.no_grad():
                for batch_idx, (images, target) in enumerate(tqdm(loader)):
                    if batch_idx < num_batches_tracked:
                        continue

                    images = images.to(device)

                    feature = self.forward(images)

                    features.append(feature.cpu())
                    targets.append(target)

                    num_batches_tracked += 1
                    # If we have reached the number of batches to cache, save the features and targets and reset the lists
                    if (num_batches_tracked % num_batches_in_cache) == 0:
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

            features_list = []
            targets_list = []

            for k in tqdm(
                range(num_cached),
                total=num_cached,
                desc=f"Loading cached features{save_str} and targets{save_str}",
            ):
                next_features = torch.load(
                    os.path.join(feature_dir, f"features{save_str}_cache_{k}.pt"),
                    map_location="cpu",
                )
                features_list.append(next_features)

                next_targets = torch.load(
                    os.path.join(feature_dir, f"targets{save_str}_cache_{k}.pt"),
                    map_location="cpu",
                )
                targets_list.append(next_targets)

            print(f"Concatenating loaded features{save_str} and targets{save_str}")
            features = torch.cat(features_list)
            targets = torch.cat(targets_list)

            for k in range(num_cached):
                os.remove(os.path.join(feature_dir, f"features{save_str}_cache_{k}.pt"))
                os.remove(os.path.join(feature_dir, f"targets{save_str}_cache_{k}.pt"))

            torch.save(features, os.path.join(feature_dir, f"features{save_str}.pt"))
            torch.save(targets, os.path.join(feature_dir, f"targets{save_str}.pt"))
