import sys
from pathlib import Path

sys.path.append('..')

from scripts.project_location import BASE_PATH_PROJECT as bpp
from scripts.project_location import RESULTS_ROOT as bpr

########################################################################
## DEFINE BASEPATHS
BASE_PATH_PROJECT = Path(bpp)
BASE_PATH_RESULTS = Path(bpr)

########################################################################
## Path to the model config file
model_config_file = '../scripts/configs/models_config_wo_alignment.json'
ds_info_file = '../scripts/configs/dataset_info.json'
ds_list_sim_file = '../scripts/configs/webdatasets_w_insub10k.txt'
ds_list_sim_file_30k = '../scripts/configs/webdatasets_w_insub30k.txt'
ds_list_perf_file = '../scripts/configs/webdatasets_w_in1k.txt'

########################################################################
## DEFINE CONSTANT LISTS

similarity_metrics = [
    'cka_kernel_rbf_unbiased_sigma_0.2',
    'cka_kernel_rbf_unbiased_sigma_0.4',
    'cka_kernel_rbf_unbiased_sigma_0.6',
    'cka_kernel_rbf_unbiased_sigma_0.8',
    'cka_kernel_linear_unbiased',
    'rsa_method_correlation_corr_method_pearson',
    'rsa_method_correlation_corr_method_spearman',
]

anchors = [
    'OpenCLIP_RN50_openai',
    'OpenCLIP_ViT-L-14_openai',
    'resnet50',
    'vit_large_patch16_224',
    'simclr-rn50',
    # 'dino-vit-base-p16',
    'dinov2-vit-large-p14',
    'mae-vit-large-p16'
]

exclude_models = []

exclude_models_w_mae = ['mae-vit-base-p16', 'mae-vit-large-p16', 'mae-vit-huge-p14']

available_data = [
    'agg_pearsonr_all_ds.csv',
    'agg_pearsonr_all_ds_with_rsa.csv',
    'agg_pearsonr_all_ds_wo_mae.csv',
    'agg_spearmanr_all_ds.csv',
    'agg_spearmanr_all_ds_with_rsa.csv',
    'agg_spearmanr_all_ds_wo_mae.csv'
]

model_categories = ['objective', 'architecture_class', 'dataset_class', 'size_class']

model_size_order = ['small', 'medium', 'large', 'xlarge']

########################################################################
## DEFINE NAME MAPPINGS
sim_metric_name_mapping = {
    'cka_kernel_rbf_unbiased_sigma_0.2': 'CKA RBF 0.2',
    'cka_kernel_rbf_unbiased_sigma_0.4': 'CKA RBF 0.4',
    'cka_kernel_rbf_unbiased_sigma_0.6': 'CKA RBF 0.6',
    'cka_kernel_rbf_unbiased_sigma_0.8': 'CKA RBF 0.8',
    'cka_kernel_linear_unbiased': 'CKA linear',
    'rsa_method_correlation_corr_method_pearson': 'RSA pearson',
    'rsa_method_correlation_corr_method_spearman': 'RSA spearman',
}

ds_name_mapping = {
    'wds_imagenet1k': 'ImageNet1k',
    'imagenet-subset-10k': 'ImageNet1k',
    'imagenet-subset-30k': 'ImageNet1k',
    'wds_vtab_flowers': 'Flowers',
    'wds_vtab_pets': 'Pets',
    'wds_vtab_eurosat': 'Eurosat',
    'wds_vtab_pcam': 'PCAM',
    'wds_vtab_dtd': 'DTD'
}

anchor_name_mapping = {
    'OpenCLIP_RN50_openai': 'OpenCLIP RN50',
    'OpenCLIP_ViT-L-14_openai': 'OpenCLIP ViT-L',
    'resnet50': 'ResNet-50',
    'vit_large_patch16_224': 'ViT-L',
    'simclr-rn50': 'SimCLR RN50',
    'dino-vit-base-p16': 'DINO ViT-B',
    'dinov2-vit-large-p14': 'DINOv2 ViT-L',
    'mae-vit-large-p16': 'MAE ViT-L'
}

# model_cat_mapping = {'objective': 'Objective', 'architecture_class': 'Architecture',
#                      'dataset_class': 'Dataset diversity',
#                      'size_class': 'Model size'}

model_cat_mapping = {'objective': 'Training objective',
                     'architecture_class': 'Architecture',
                     'dataset_class': 'Training data',
                     'size_class': 'Model size'}

model_ca_orig_mapping = {v: k for k, v in model_cat_mapping.items()}

cat_name_mapping = {
    'All': 'All',
    'Image-Text': 'Img-Txt',
    'Self-Supervised': 'SSL',
    'Self-Supervised-2': 'SSL-2',
    'Self-Supervised-3': 'SSL-3',
    'Self-Supervised-4': 'SSL-4',
    'Supervised': 'Sup',
    'Large DS': 'Large',
    'XLarge DS': 'XLarge',
    'Large': 'Large',
    'XLarge': 'XLarge',
    'ImageNet1k': 'IN1k',
    'ImageNet21k': 'IN21k',
    'Convolutional': 'CNN',
    'Transformer': 'TX',
    'small': 'small',
    'xlarge': 'xlarge',
    'medium': 'medium',
    'large': 'large',
}

########################################################################
## PLOTTING CONSTANTS
fontsizes = {
    'title': 14,
    'legend': 13,
    'label': 13,
    'ticks': 12,
}

fontsizes_cols = {
    'title': 18,
    'legend': 17,
    'label': 17,
    'ticks': 16,
}

cat_color_mapping = {'Img-Txt': '#1f77b4',
                     'SSL': '#ff7f0e',
                     'Sup': '#2ca02c',
                     'CNN': '#d62728',
                     'TX': '#9467bd',
                     'IN1k': '#8c564b',
                     'IN21k': '#e377c2',
                     'Large DS': '#7f7f7f',
                     'XLarge DS': '#bcbd22',
                     'Large': '#7f7f7f',
                     'XLarge': '#bcbd22',
                     'small': '#17becf',
                     'medium': '#66c2a5',
                     'large': '#fc8d62',
                     'xlarge': '#8da0cb'}

cm = 0.393701
