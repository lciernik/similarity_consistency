sim_metric_name_mapping = {
    'cka_kernel_rbf_unbiased_sigma_0.2': 'CKA RBF 0.2',
    'cka_kernel_rbf_unbiased_sigma_0.4': 'CKA RBF 0.4',
    'cka_kernel_rbf_unbiased_sigma_0.6': 'CKA RBF 0.6',
    'cka_kernel_rbf_unbiased_sigma_0.8': 'CKA RBF 0.8',
    'cka_kernel_linear_unbiased': 'CKA linear',
    'rsa_method_correlation_corr_method_pearson': 'RSA pearson',
    'rsa_method_correlation_corr_method_spearman': 'RSA spearman',
}

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
    'dino-vit-base-p16',
    'dinov2-vit-large-p14',
    'mae-vit-large-p16'
]

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

exclude_models = []

exclude_models_w_mae = ['mae-vit-base-p16', 'mae-vit-large-p16', 'mae-vit-huge-p14']

available_data = [
    # 'agg_spearmanr_all_ds_wo_swav_pirl_timm_clip.csv',
    # 'agg_spearmanr_all_ds_wo_mae_swav_pirl_timm_clip.csv',
    # 'agg_pearsonr_all_ds_wo_swav_pirl_timm_clip.csv',
    # 'agg_pearsonr_all_ds_wo_mae_swav_pirl_timm_clip.csv',
    'agg_pearsonr_all_ds.csv',
    'agg_pearsonr_all_ds_wo_mae.csv',
    'agg_spearmanr_all_ds.csv',
    'agg_spearmanr_all_ds_wo_mae.csv'
]

ds_name_mapping = {
    'wds_imagenet1k': 'ImageNet1k',
    'imagenet-subset-10k': 'ImageNet1k',
    'wds_vtab_flowers': 'Flowers',
    'wds_vtab_pets': 'Pets',
    'wds_vtab_eurosat': 'Eurosat',
    'wds_vtab_pcam': 'PCAM'
}

model_categories = ['objective', 'architecture_class', 'dataset_class', 'size_class']
model_cat_mapping = {'objective': 'Objective', 'architecture_class': 'Architecture', 'dataset_class': 'Dataset diversity',
                     'size_class': 'Model size'}
model_ca_orig_mapping = {v:k for k,v in model_cat_mapping.items()}

model_size_order = ['small', 'medium', 'large', 'xlarge']

cat_name_mapping = {
    'All': 'All',
    'Image-Text': 'Img-Txt',
    'Self-Supervised': 'SSL',
    'Supervised': 'Sup',
    'Large DS': 'Large DS',
    'XLarge DS': 'XLarge DS',
    'ImageNet1k': 'IN1k',
    'ImageNet21k': 'IN21k',
    'Convolutional': 'CNN',
    'Transformer': 'TX',
    'small': 'small',
    'xlarge': 'xlarge',
    'medium': 'medium',
    'large': 'large',
}

model_config_file = '../scripts/models_config_wo_alignment.json'

ds_info_file = '../scripts/dataset_info.json'

## FONTSIZES 
fontsizes = {
    'title': 13,
    'legend': 12,
    'label': 12,
    'ticks': 11,
}




