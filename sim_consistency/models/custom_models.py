from pathlib import Path
import re
import torch
import torchvision.models as models
from thingsvision.custom_models.custom import Custom
import torch.nn as nn

PRETRAINED_MODEL_PATH = Path("/home/space/diverse_priors/pretrained_models")

CUSTOM_MODELS = [
    "alexnet_places365",
    "densenet161_places365",
    "resnet18_places365",
    "resnet50_places365",
]


class CustomModelPlaces(Custom):
    def __init__(self, model_name: str, device: str) -> None:
        super().__init__(device)
        self.backend = "pt"
        self.model_name = model_name

    def create_model(self):
        arch, _ = self.model_name.split("_")
        model_file = PRETRAINED_MODEL_PATH / f"{self.model_name}.pth.tar"
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {
            str.replace(k, "module.", ""): v
            for k, v in checkpoint["state_dict"].items()
        }
        if self.model_name == "densenet161_places365":
            state_dict = {
                re.sub(r"\.(\d+)\.", r"\1.", k): v for k, v in state_dict.items()
            }

        model.load_state_dict(state_dict)
        return model
