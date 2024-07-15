import argparse
import os
from typing import List, Tuple, Optional

from clip_benchmark.utils.utils import as_list, all_paths_exist


class PathMaker:
    def __init__(self, args: argparse.Namespace, dataset_name: str, probe_dataset_name: Optional[str] = None, auto_create_dirs: bool = True):
        self.dataset_name = dataset_name
        self.train_dataset_name = probe_dataset_name if probe_dataset_name is not None else dataset_name
        self.task = args.task
        self.mode = args.mode
        
        self.auto_create_dirs = auto_create_dirs

        self.dataset_root = args.dataset_root
        self.feature_root = args.feature_root
        self.model_root = args.model_root
        self.output_root = args.output_root
        self._check_root_paths()

        self.model_ids = as_list(args.model_key)
        self.feature_combiner = args.feature_combiner
        self.hyperparams_slug = self._get_hyperparams_name(args)
        self.model_slug = self._create_model_slug()

        self.verbose = args.verbose

    @staticmethod
    def _get_hyperparams_name(args: argparse.Namespace) -> str:
        """Get the hyperparameters name for the output path."""
        fewshot_slug = "no_fewshot" if args.fewshot_k == -1 else f"fewshot_{args.fewshot_k}"
        subpath = os.path.join(fewshot_slug,
                               f"fewshot_lr_{args.fewshot_lr}",
                               f"fewshot_epochs_{args.fewshot_epochs}",
                               f"batch_size_{args.batch_size}",
                               f"seed_{args.seed}",
                               )
        return subpath

    def _create_model_slug(self) -> str:
        model_slug = '__'.join(self.model_ids)
        if self.task == "linear_probe" and self.mode == "combined_models":
            model_slug += f"_{self.feature_combiner}"
        return model_slug

    def _check_root_paths(self) -> None:
        """Check existence of the feature, model and output folders."""
        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError(f"Dataset root folder {self.dataset_root} does not exist.")
        if not os.path.exists(self.feature_root):
            raise FileNotFoundError(f"Feature root folder {self.feature_root} does not exist.")
        if not os.path.exists(self.model_root) and self.task == "linear_probe":
            raise FileNotFoundError(f"Model root folder {self.model_root} does not exist.")
        if not os.path.exists(self.output_root) and self.task == "linear_probe" and self.auto_create_dirs:
            os.makedirs(self.output_root, exist_ok=True)
            if self.verbose:
                print(f'Created path ({self.output_root}), where results are to be stored ...')

    def _get_feature_dirs(self) -> List[str]:
        feature_dirs = [os.path.join(self.feature_root, self.dataset_name, model_id) for model_id in self.model_ids]
        if self.task == "linear_probe" and self.mode != "single_model":
            if not all_paths_exist(feature_dirs):
                raise FileNotFoundError(
                    f"Not all feature directories exist: {feature_dirs}. Cannot evaluate linear probe with multiple models. "
                    f"Run the linear probe for each model separately first."
                )
        return feature_dirs

    def _get_model_dirs(self) -> List[str]:
        if self.task == "linear_probe" and self.mode == "combined_models":
            model_dirs = [
                os.path.join(self.model_root, self.train_dataset_name, self.model_slug, self.hyperparams_slug)]
        else:
            model_dirs = [os.path.join(self.model_root, self.train_dataset_name, model_id, self.hyperparams_slug)
                          for model_id in self.model_ids]
        return model_dirs

    def _get_results_and_predictions_dirs(self) -> str:
        base_results_dir = self.get_base_results_dir()
        results_dir = os.path.join(base_results_dir, self.hyperparams_slug)

        if not os.path.exists(results_dir) and self.auto_create_dirs:
            os.makedirs(results_dir, exist_ok=True)
            if self.verbose:
                print(f'Created path ({results_dir}), where results are to be stored ...')

        return results_dir

    def get_base_results_dir(self) -> str:
        base_results_dir = os.path.join(self.output_root, self.task, self.mode, self.dataset_name, self.model_slug)
        return base_results_dir

    def _get_single_prediction_dirs(self) -> List[str]:
        single_prediction_dirs = [os.path.join(self.output_root, self.task, 'single_model', self.dataset_name, model_id,
                                               self.hyperparams_slug)
                                  for model_id in self.model_ids
                                  ]
        if not all_paths_exist(single_prediction_dirs):
            raise FileNotFoundError(
                f"Not all single prediction directories exist: {single_prediction_dirs}. "
                f"Cannot evaluate ensemble model."
            )
        return single_prediction_dirs

    def make_paths(self) -> Tuple[List[str], Optional[List[str]], Optional[str], Optional[List[str]], List[str]]:
        feature_dirs = self._get_feature_dirs()

        if self.task == "feature_extraction":
            return feature_dirs, None, None, None, self.model_ids

        model_dirs = self._get_model_dirs()
        results_dir = self._get_results_and_predictions_dirs()

        if self.task == "linear_probe" and self.mode == "ensemble":
            single_prediction_dirs = self._get_single_prediction_dirs()
        else:
            single_prediction_dirs = None

        return feature_dirs, model_dirs, results_dir, single_prediction_dirs, self.model_ids
