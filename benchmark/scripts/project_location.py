import os

## DEFINE BASEPATHS
BASE_PATH_PROJECT = '/home/space/diverse_priors'

### DEFINE SUBFOLDERS

DATASETS_ROOT = os.path.join(BASE_PATH_PROJECT, 'datasets')

SUBSET_ROOT = os.path.join(DATASETS_ROOT, 'subsets')

FEATURES_ROOT = os.path.join(BASE_PATH_PROJECT, 'features')

MODELS_ROOT = os.path.join(BASE_PATH_PROJECT, 'models')

MODEL_SIM_ROOT = os.path.join(BASE_PATH_PROJECT, 'model_similarities')

RESULTS_ROOT = os.path.join(BASE_PATH_PROJECT, 'results')
