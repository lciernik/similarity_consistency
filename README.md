# Diverse Priors

Can we combine representations of different models to improve label efficiency and robustness?

## Getting Started

To setup the feature extraction and benchmark:

```bash
cd benchmark
pip install .
```

Scripts to run feature extraction and evaluation of combined models are found in `benchmark/scripts`.
To check if you set up everything correctly, you can run `./benchmark/scripts/test_scripts/test_single.sh`.

## Project folder structure

Base folder located at `/home/space/diverse_priors`. It hase the following structure:

- `clustering`: Contains the `cluster_label.csv` files for each dataset, model and number of cluster (k).
- `datasets`: Contains the datasets used in the project. We mainly use the `imagenet-subset-xxk` and `wds` datasets.
- `features`: Contains the extracted features for each dataset and model.E.g.,
    - `features/imagenet-subset-10k/dinov2-vit-large-p14` contains the
      files `features_test.pt`, `features_train.pt`, `targets_test.pt`, and `targets_train.pt`.
- `model_similarities`: Directory to store the similarity matrices computed with different metrics between models.
    - E.g., `model_similarities/imagenet-subset-10k/cka_kernel_linear_unbiased` contains the similarity matrix between
      the features of the models in `benchmark/scripts/model_config.json` computed with the CKA metric with linear
      kernel.
- `models`: Contains the trained linear probes of each dataset and model for single and combined models, but not
  ensembles.
- `results`:
    - It contains subfolders, e.g., `results/imagenet-subset-10k/dinov2-vit-large-p14/[HYPERPARAMETER]`
      holding the `test_predictions.csv` and the `results.json` files.
- `sampling`: Contains the `sampling.csv` files for each dataset, model and number of cluster (k).

## Run Commands

```bash 
python feature_extraction.py --models_config ./models_config.json --datasets "wds/imagenet1k wds/imagenetv2 wds/imagenet-a wds/imagenet-r wds/imagenet_sketch"
python single_model_evaluation.py --models_config ./models_config.json --datasets "wds/imagenet1k wds/imagenetv2 wds/imagenet-a wds/imagenet-r wds/imagenet_sketch"
```

Ensemble:

```bash 
python combined_models_evaluation.py --models_config ./models_config.json  --sampling_folder  models_3-samples_10  --datasets "wds/imagenetv2 wds/imagenet-a wds/imagenet-r wds/imagenet_sketch"
```

Combined models:

```bash
python combined_models_evaluation.py --combination concat --models_config ./models_config.json  --sampling_folder  models_3-samples_10 --datasets "wds/imagenet1k wds/imagenetv2 wds/imagenet-a wds/imagenet-r wds/imagenet_sketch"
```

## Sampling

We want to test different model selection strategies.
Let's assume we want to sample `num_samples` model sets of size `k`.
Then, we can do that with the following command.

```bash
python sim_consistency/sample_models.py \
  --num_models <k> \
  --sampling strategies top-k random cluster one-cluster \
  --num_samples <num_samples> \
  --output_root /home/space/diverse_priors/sampling/models_<k>-samples_<num_samples>
```

This will write a json with the sampled model sets for each of the sampling strategies into
`/home/space/diverse_priors/sampling/models_<k>-samples_<num_samples>`.

To run sampling for multiple model set sizes, we can use `benchmark/scripts/run_sampling.py` \
which will automatically trigger the necessary jobs.

