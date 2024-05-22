import argparse
from typing import Tuple, List


def get_parser_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """Get the parser arguments."""
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        return parser.add_argument(*args, **kwargs)

    # DATASET
    aa('--dataset', type=str, default="cifar10", nargs="+",
       help="Dataset(s) to use for the benchmark. Can be the name of a dataset, or a collection "
            "name ('vtab', 'vtab+', 'imagenet_robustness', 'retrieval') or path of a text file "
            "where each line is a dataset name")
    aa('--dataset_root', default="root", type=str,
       help="dataset root folder where the data are downloaded. Can be in the form of a "
            "template depending on dataset name, e.g., --dataset_root='data/{dataset}'. "
            "This is useful if you evaluate on multiple data.")
    aa('--split', type=str, default="test", help="Dataset split to use")
    aa('--test_split', dest="split", action='store', type=str, default="test",
       help="Dataset split to use")
    aa('--train_split', type=str, nargs='+', default="train",
       help="Dataset(s) train split names")
    mutually_exclusive = parser.add_mutually_exclusive_group()
    mutually_exclusive.add_argument('--val_split', default=None, type=str, nargs="+",
                                    help="Dataset(s) validation split names. Mutually exclusive with val_proportion.")
    mutually_exclusive.add_argument('--val_proportion', default=None, type=float, nargs="+",
                                    help="what is the share of the train dataset will be used for validation part, "
                                         "if it doesn't predefined. Mutually exclusive with val_split")
    aa('--wds_cache_dir', default=None, type=str,
       help="optional cache directory for webdataset only")
    aa('--custom_classname_file', default=None, type=str,
       help="use custom json file with classnames for each dataset, where keys are dataset "
            "names and values are list of classnames.")

    # FEATURES
    aa('--feature_root', default="features", type=str,
       help="feature root folder where the features are stored.")
    # TODO: change alignment to argument such that it can be model specific, b/c some model do not have alignment.
    aa('--feature_alignment', nargs='?', const='gLocal',
       type=lambda x: None if x == '' else x)
    aa('--normalize', dest='normalize', action="store_true", default=True,
       help="enable features normalization")
    aa('--no-normalize', dest='normalize', action='store_false',
       help="disable features normalization")

    # MODEL(S)
    aa('--model', type=str, nargs="+", default=["dinov2-vit-large-p14"],
       help="Thingsvision model string")
    aa('--model_source', type=str, nargs="+", default=["ssl"],
       help="For each model, indicate the source of the model. "
            "See thingsvision for more details.")
    aa('--model_parameters', nargs="+", type=str,
       help='A serialized JSON dictionary of key-value pairs.')
    aa('--module_name', type=str, nargs="+", default=["norm"], help="Module name")

    # TASKS
    aa('--task', type=str, default="linear_probe",
       choices=["linear_probe", "model_similarity"],
       help="Task to evaluate on. With --task=auto, the task is automatically inferred from the "
            "dataset.")
    aa('--mode', type=str, default="single_model",
       choices=["single_model", "combined_models", "ensemble"],
       help="Mode to use for linear probe task.")
    aa('--eval_combined', action="store_true",
       help="Whether the features of the different models should be used in combined fashion.")
    aa('--feature_combiner', type=str, default="concat",
       choices=['concat', 'concat_pca'], help="Feature combiner to use")

    aa('--fewshot_k', default=[-1], type=int, nargs="+",
       help="for linear probe, how many shots. -1 = whole dataset.")
    aa('--fewshot_epochs', default=[10], type=int, nargs='+',
       help="for linear probe, how many epochs.")
    aa('--fewshot_lr', default=[0.1], type=float, nargs='+',
       help="for linear probe, what is the learning rate.")
    aa('--batch_size', default=64, type=int)
    aa('--no_amp', action="store_false", dest="amp", default=True,
       help="whether to use mixed precision")
    aa("--skip_load", action="store_true",
       help="for linear probes, when everything is cached, no need to load model.")
    aa('--skip_existing', default=False, action="store_true",
       help="whether to skip an evaluation if the output file exists.")

    ### Model similarity
    aa('--sim_method', type=str, default="cka",
       choices=['cka', 'rsa'], help="Method to use for model similarity task.")
    aa('--sim_kernel', type=str, default="linear",
       choices=['linear'], help="Kernel used during CKA. Ignored if sim_method is rsa.")
    aa('--rsa_method', type=str, default="correlation",
       choices=['cosine', 'correlation'],
       help="Method used during RSA. Ignored if sim_method is cka.")
    aa('--corr_method', type=str, default="spearman",
       choices=['pearson', 'spearman'],
       help="Kernel used during CKA. Ignored if sim_method is cka.")
    aa('--sigma', type=float, default=None, help="sigma for CKA rbf kernel.")
    aa('--biased_cka', action="store_false", dest="unbiased", help="use biased CKA")

    # STORAGE
    aa('--output_root', default="results", type=str,
       help="Path to root folder where the results are stored.")
    aa('--model_root', default="models", type=str,
       help="Path to root folder where linear probe model checkpoints are stored.")

    # GENERAL
    aa('--num_workers', default=4, type=int)

    aa("--distributed", action="store_true", help="evaluation in parallel")
    aa('--quiet', dest='verbose', action="store_false",
       help="suppress verbose messages")

    # REPRODUCABILITY
    aa('--seed', default=[0], type=int, nargs='+', help="random seed.")

    args = parser.parse_args()
    return parser, args


def prepare_args(args: argparse.Namespace, model_info: Tuple[str, str, dict, str]) -> argparse.Namespace:
    args.model = model_info[0]  # model
    args.model_source = model_info[1]  # model_source
    args.model_parameters = model_info[2]  # model_parameters
    args.module_name = model_info[3]  # module_name
    return args


def prepare_combined_args(args: argparse.Namespace, model_comb: List[Tuple[str, str, dict, str]]) -> argparse.Namespace:
    args.model = [tup[0] for tup in model_comb]
    args.model_source = [tup[1] for tup in model_comb]
    args.model_parameters = [tup[2] for tup in model_comb]
    args.module_name = [tup[3] for tup in model_comb]
    return args