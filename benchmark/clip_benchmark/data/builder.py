import json
import os
from subprocess import call
import webdataset as wds

import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.data import default_collate
from torchvision.datasets import ImageNet

from clip_benchmark.data.constants import all_imagenet_wordnet_ids, dataset_collection
from clip_benchmark.data.datasets import breeds, cifar_coarse, imagenet_variants


def build_dataset(dataset_name, root="root", transform=None, split="test", download=True, wds_cache_dir=None):
    """
    Main function to use in order to build a dataset instance,

    dataset_name: str
        name of the dataset
    
    root: str
        root folder where the dataset is downloaded and stored. can be shared among data.

    transform: torchvision transform applied to images

    split: str
        split to use, depending on the dataset can have different options.
        In general, `train` and `test` are available.
        For specific splits, please look at the corresponding dataset.
    """

    if split == "train" and any([imgnt_variant in dataset_name for imgnt_variant in
         ["imagenet-a", "imagenet-o", "imagenet-r", "imagenet_sketch", "imagenetv2"]]):
        # Setting imagenet as training set for imagenet variants (should not be used)
        root = os.sep.join(root.split(os.sep)[:-1] + ["wds_imagenet1k"])
        print(f"Using imagenet1k as train dataset for {dataset_name}, dataset root set to {root}")
        dataset_name = "wds/imagenet1k"

    if dataset_name == "imagenet-subset-10k":
        root = os.path.join(root, 'imagenet_torch')

        if not os.path.exists(root) and download:
            os.makedirs(root, exist_ok=True)
            call(
                    f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --output-document={r}/ILSVRC2012_devkit_t12.tar.gz",
                    shell=True)
            call(
                    f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --output-document={r}/ILSVRC2012_img_train.tar",
                    shell=True)
            call(
                    f"wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --output-document={r}/ILSVRC2012_img_val.tar",
                    shell=True)

        if split == 'test':
            print('There is no test split, using val split.')
            split = 'val'
        ds_in = ImageNet(root=root, split=split, transform=transform)
        with open(os.path.join(root, f'imagenet-10k-{split}.json'), 'r') as f:
            indices_map = json.load(f)
        indices = np.array(list(map(list, indices_map.values()))).flatten()
        ds = Subset(ds_in, indices=indices)
        ds.classes = ds_in.classes
        if ds is None:
            print("imagenet-subset-10k dataset build did not work")
        else:
            print("successfully build imagenet-subset-10k dataset")
    elif dataset_name == "cifar100-coarse":
        root = "/".join(root.split("/")[:-1] + ["wds_vtab-cifar100"])
        superclass_mapping, superclass_names = cifar_coarse.get_cifar100_coarse_map()
        ds = build_wds_dataset(transform=transform, split=split, data_dir=root,
                               cache_dir=wds_cache_dir,
                               label_map=lambda y: superclass_mapping[y],
                               )
        ds.classes = superclass_names
    elif dataset_name in breeds.get_breeds_task_names():
        root = "/".join(root.split("/")[:-1] + ["wds_imagenet1k"])
        train_classes, test_classes, superclass_mapping = breeds.get_breeds_task(dataset_name)
        classes = train_classes if split == "train" else test_classes
        ds = build_wds_dataset(transform=transform, split=split, data_dir=root, cache_dir=wds_cache_dir,
                               selector_fn=lambda x: int(x["cls"]) in classes,
                               label_map=lambda y: superclass_mapping[y]
                               )
    elif dataset_name.startswith("wds/"):
        ds = build_wds_dataset(transform=transform, split=split, data_dir=root, cache_dir=wds_cache_dir)
    elif dataset_name == "dummy":
        ds = Dummy()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}.")

    return ds


class Dummy():

    def __init__(self):
        self.classes = ["blank image", "noisy image"]

    def __getitem__(self, i):
        return torch.zeros(3, 224, 224), 0

    def __len__(self):
        return 1


def get_dataset_default_task(dataset):
    if dataset in (
            "flickr30k", "flickr8k", "mscoco_captions", "multilingual_mscoco_captions", "flickr30k-200",
            "crossmodal3600",
            "xtd200"):
        return "zeroshot_retrieval"
    elif dataset.startswith("sugar_crepe") or dataset == "winoground":
        return "image_caption_selection"
    else:
        return "zeroshot_classification"


def get_dataset_collate_fn(dataset_name):
    if dataset_name in (
            "mscoco_captions", "multilingual_mscoco_captions", "flickr30k", "flickr8k", "flickr30k-200",
            "crossmodal3600",
            "xtd200", "winoground") or dataset_name.startswith("sugar_crepe"):
        return image_captions_collate_fn
    else:
        return default_collate


def get_dataset_class_filter(dataset_name):
    class_filter = None
    if any([n in dataset_name for n in ("imagenet-r", "imagenet-a")]):
        classes = imagenet_variants.get_class_ids(dataset_name.replace("wds/", ""))
        class_filter = torch.eye(1000, dtype=bool)[[classes]]
    return class_filter


def has_gdown():
    return call("which gdown", shell=True) == 0


def has_kaggle():
    return call("which kaggle", shell=True) == 0


def build_wds_dataset(transform, split="test", data_dir="root", cache_dir=None, selector_fn=None,
                      label_map=None):
    """
    Load a dataset in WebDataset format. Either local paths or HTTP URLs can be specified.
    Expected file structure is:
    ```
    data_dir/
        train/
            nshards.txt
            0.tar
            1.tar
            ...
        test/
            nshards.txt
            0.tar
            1.tar
            ...
        classnames.txt
        zeroshot_classification_templates.txt
        dataset_type.txt
    ```
    Classnames and templates are required for zeroshot classification, while dataset type
    (equal to "retrieval") is required for zeroshot retrieval data.

    You can use the `clip_benchmark_export_wds` or corresponding API
    (`clip_benchmark.webdataset_builder.convert_dataset`) to convert data to this format.

    Set `cache_dir` to a path to cache the dataset, otherwise, no caching will occur.
    """

    def read_txt(fname):
        if "://" in fname:
            stream = os.popen("curl -L -s --fail '%s'" % fname, "r")
            value = stream.read()
            if stream.close():
                raise FileNotFoundError("Failed to retreive data")
        else:
            with open(fname, "r") as file:
                value = file.read()
        return value

    # Special handling for Huggingface data
    # Git LFS files have a different file path to access the raw data than other files
    if data_dir.startswith("https://huggingface.co/datasets"):
        # Format: https://huggingface.co/datasets/<USERNAME>/<REPO>/tree/<BRANCH>
        *split_url_head, _, url_path = data_dir.split("/", 7)
        url_head = "/".join(split_url_head)
        metadata_dir = "/".join([url_head, "raw", url_path])
        tardata_dir = "/".join([url_head, "resolve", url_path])
    else:
        metadata_dir = tardata_dir = data_dir
    # Get number of shards
    nshards_fname = os.path.join(metadata_dir, split, "nshards.txt")
    nshards = int(read_txt(nshards_fname))  # Do not catch FileNotFound, nshards.txt should be mandatory
    # Get dataset type (classification or retrieval)
    type_fname = os.path.join(metadata_dir, "dataset_type.txt")
    try:
        dataset_type = read_txt(type_fname).strip().lower()
    except FileNotFoundError:
        # print("WARNING: dataset_type.txt not found, assuming type=classification")
        dataset_type = "classification"
    #
    filepattern = os.path.join(tardata_dir, split, "{0..%d}.tar" % (nshards - 1))
    # Load webdataset (support WEBP, PNG, and JPG for now)
    if not cache_dir or not isinstance(cache_dir, str):
        cache_dir = None
    if selector_fn is None:
        dataset = (
            wds.WebDataset(filepattern, cache_dir=cache_dir, nodesplitter=lambda src: src)
            .decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"]))
        )
    else:
        dataset = (
            wds.WebDataset(filepattern, cache_dir=cache_dir, nodesplitter=lambda src: src).select(selector_fn)
            .decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"]))
        )
    # Load based on classification or retrieval task
    if dataset_type == "retrieval":
        dataset = (dataset
                   .to_tuple(["webp", "png", "jpg", "jpeg"], "txt")
                   .map_tuple(transform, str.splitlines)
                   )
        dataset.classes = dataset.templates = None
    else:
        label_type = "npy" if dataset_type == "multilabel" else "cls"  # Special case for multilabel
        dataset = (dataset
                   .to_tuple(["webp", "png", "jpg", "jpeg"], label_type)
                   .map_tuple(transform, label_map)
                   )
        # Get class names if present
        classnames_fname = os.path.join(metadata_dir, "classnames.txt")
        try:
            dataset.classes = [line.strip() for line in read_txt(classnames_fname).splitlines()]
        except FileNotFoundError:
            print("WARNING: classnames.txt not found")
            dataset.classes = None

    return dataset


def _extract_task(dataset_name):
    prefix, *task_name_list = dataset_name.split("_")
    task = "_".join(task_name_list)
    return task


def image_captions_collate_fn(batch):
    transposed = list(zip(*batch))
    imgs = default_collate(transposed[0])
    texts = transposed[1]
    return imgs, texts


def get_dataset_collection_from_file(path):
    return [l.strip() for l in open(path).readlines()]


def get_dataset_collection():
    return dataset_collection
