import zipfile
import zlib
from typing import Union, Dict, Optional

import torch
from thingsvision import get_extractor


class ThingsvisionModel:
    def __init__(self, extractor, module_name, feature_alignment=None, flatten_acts=False):
        self._extractor = extractor
        self._module_name = module_name
        self._extractor.model = self._extractor.model.to(extractor.device)
        self._extractor.activations = {}
        self._output_type = "tensor"
        self._alignment_type = feature_alignment
        self._flatten_acts = flatten_acts

    def encode_image(self, x):
        with self._extractor.batch_extraction(
                self._module_name, output_type=self._output_type
        ) as e:
            features = e.extract_batch(
                batch=x,
                flatten_acts=self._flatten_acts,
                # flatten 2D feature maps from an early convolutional or attention layer
            )
            if len(features.shape) == 3:
                model_name = self._extractor.model_name
                model_parameters = self._extractor.model_parameters
                if model_name.startswith("OpenCLIP"):
                    if "variant" in model_parameters and not self._module_name == "visual":
                        if "SigLIP" not in model_parameters["variant"] and "ViT" in model_parameters["variant"]:
                            features = features[0]

        features = features.to(torch.float32)

        if self._alignment_type is not None:
            is_aligned = False
            while not is_aligned:
                try:
                    features = self._extractor.align(
                        features=features,
                        module_name=self._module_name,
                        alignment_type=self._alignment_type,
                    )
                    is_aligned = True
                except (zipfile.BadZipFile, FileNotFoundError, EOFError, zlib.error, ValueError) as e:
                    print(f"Error: {e}", flush=True)

        return features

    def parameters(self):
        return self._extractor.model.parameters()

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


def load_thingsvision_model(
        model_name: str,
        source: str,
        device: Union[str, torch.device],
        model_parameters: Dict,
        module_name: str,
        feature_alignment: Optional[str] = None,
):
    extractor = get_extractor(
        model_name=model_name,
        source=source,
        device=device,
        pretrained=True,
        model_parameters=model_parameters,
    )

    flatten_acts = True
    if (
            "extract_cls_token" in model_parameters
            and model_parameters["extract_cls_token"]
            or "token_extraction" in model_parameters
    ):
        flatten_acts = False
    elif model_name.startswith("OpenCLIP"):
        if "variant" in model_parameters and not module_name == "visual":
            if "SigLIP" not in model_parameters["variant"] and "ViT" in model_parameters["variant"]:
                flatten_acts = False

    model = ThingsvisionModel(
        extractor=extractor,
        module_name=module_name,
        feature_alignment=feature_alignment,
        flatten_acts=flatten_acts,
    )
    transform = extractor.get_transformations(resize_dim=256, crop_dim=224)
    return model, transform
