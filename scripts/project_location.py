import os

## DEFINE BASEPATHS
BASE_PATH_PROJECT = ''  # TODO: Add the path to the project folder

### DEFINE SUBFOLDERS

DATASETS_ROOT = os.path.join(BASE_PATH_PROJECT, 'datasets')

SUBSET_ROOT = os.path.join(DATASETS_ROOT, 'subsets_30k')

FEATURES_ROOT = os.path.join(BASE_PATH_PROJECT, 'features')

MODELS_ROOT = os.path.join(BASE_PATH_PROJECT, 'models')

MODEL_SIM_ROOT = os.path.join(BASE_PATH_PROJECT, 'model_similarities')

RESULTS_ROOT = os.path.join(BASE_PATH_PROJECT, 'results_rebuttal')

if __name__ == "__main__":
    paths = [
        DATASETS_ROOT,
        SUBSET_ROOT,
        FEATURES_ROOT,
        MODELS_ROOT,
        MODEL_SIM_ROOT,
        RESULTS_ROOT
    ]
    if not BASE_PATH_PROJECT:
        raise ValueError("Please set the BASE_PATH_PROJECT variable in project_location.py to the project folder path.")

    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")
