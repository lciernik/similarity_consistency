from thingsvision import get_extractor
import torch


class ThingsvisionModel:

    def __init__(self, extractor, module_name):
        self._extractor = extractor
        self._module_name = module_name
        self._extractor.model = self._extractor.model.to(extractor.device)
        self._extractor.activations = {}
        self._extractor.register_hook(module_name=module_name)

    def encode_image(self, x):
        features = self._extractor._extract_batch(
            batch=x,
            module_name=self._module_name,
            flatten_acts=True,
        )
        return features.to(torch.float32)

    def cleanup(self):
        self._extractor._unregister_hook()


def load_thingsvision_model(model_name, source, device, model_parameters, module_name):
    extractor = get_extractor(
        model_name=model_name,
        source=source,
        device=device,
        pretrained=True,
        model_parameters=model_parameters,
    )
    model = ThingsvisionModel(extractor=extractor,
                              module_name=module_name)
    transform = extractor.get_transformations(resize_dim=256, crop_dim=224)
    return model, transform
