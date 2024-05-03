import torch
import torch.nn.functional as F


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
