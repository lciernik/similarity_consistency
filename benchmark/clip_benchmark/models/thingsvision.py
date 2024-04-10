from typing import Union, Dict, Optional

import torch
import zipfile
from thingsvision import get_extractor


class ThingsvisionModel:

    def __init__(self, extractor, module_name, feature_alignment=None):
        self._extractor = extractor
        self._module_name = module_name
        self._extractor.model = self._extractor.model.to(extractor.device)
        self._extractor.activations = {}
        self._output_type = "tensor"
        self._alignment_type = feature_alignment

    def encode_image(self, x):
        with self._extractor.batch_extraction(self._module_name, output_type=self._output_type) as e:
            features = e.extract_batch(
                batch=x,
                flatten_acts=True,  # flatten 2D feature maps from an early convolutional or attention layer
            )

        features = features.to(torch.float32)
        
        if self._alignment_type is not None:
            is_aligned=False
            while not is_aligned:
                try: 
                    features = self._extractor.align(
                        features=features,
                        module_name=self._module_name,
                        alignment_type=self._alignment_type,
                    )
                    is_aligned = True
                except (zipfile.BadZipFile, FileNotFoundError, EOFError) as e:
                    print(f"Error: {e}", flush=True)

        return features


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
    model = ThingsvisionModel(extractor=extractor,
                              module_name=module_name,
                              feature_alignment=feature_alignment)
    transform = extractor.get_transformations(resize_dim=256, crop_dim=224)
    return model, transform
