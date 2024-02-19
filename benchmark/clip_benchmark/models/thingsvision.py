from thingsvision import get_extractor


class ThingsvisionModel:

    def __init__(self, extractor, module_name):
        self._extractor = extractor
        self._module_name = module_name

    def __call__(self, x):
        features = self._extractor.extract_features(
            batches=[x],
            module_name=self._module_name,
            flatten_acts=True,
            output_type="tensor",  # or "tensor" (only applicable to PyTorch models of which CLIP is one!)
        )
        return features


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
