import os

## DEFINE BASEPATHS
# BASE_PATH_PROJECT = '/home/space/diverse_priors'
BASE_PATH_PROJECT = '/home/lciernik/projects/divers-priors/results_local/test_pipeline_061124'

### DEFINE SUBFOLDERS

DATASETS_ROOT = os.path.join(BASE_PATH_PROJECT, 'datasets')

SUBSET_ROOT = os.path.join(DATASETS_ROOT, 'subsets')

FEATURES_ROOT = os.path.join(BASE_PATH_PROJECT, 'features')

MODELS_ROOT = os.path.join(BASE_PATH_PROJECT, 'models')

MODEL_SIM_ROOT = os.path.join(BASE_PATH_PROJECT, 'model_similarities')

RESULTS_ROOT = os.path.join(BASE_PATH_PROJECT, 'results')


if __name__ == "__main__":
    # Check if all paths exist and create them if not
    paths = [
        DATASETS_ROOT,
        SUBSET_ROOT,
        FEATURES_ROOT,
        MODELS_ROOT,
        MODEL_SIM_ROOT,
        RESULTS_ROOT
    ]

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

