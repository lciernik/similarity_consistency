from typing import Union, Dict, Optional
import torch
from .thingsvision import load_thingsvision_model


def load_model(
        source: str,
        model_name: str,
        module_name: str,
        model_parameters: Union[Dict, None] = None,
        feature_alignment: Optional[str] = None,
        device: Union[str, torch.device] = "cuda"
):
    return load_thingsvision_model(
        model_name=model_name,
        source=source,
        model_parameters=model_parameters,
        device=device,
        module_name=module_name,
        feature_alignment=feature_alignment
    )
