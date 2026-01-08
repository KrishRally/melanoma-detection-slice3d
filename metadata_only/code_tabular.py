import os
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import warnings
warnings.filterwarnings("ignore")
import cv2

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from sklearn.utils import resample

import optuna

from tqdm import tqdm


import os
import gc
import cv2
import math
import copy
import time
import random
import glob
from matplotlib import pyplot as plt

import h5py
from PIL import Image
from io import BytesIO

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torchvision

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# For Image Models
import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2
train_path = 'C:/Users/YASH/data/train-metadata.csv'
test_path = 'C:/Users/YASH/data/test-metadata.csv'

err = 1e-5
sampling_ratio = 0.01
seed = 42

id_col = 'isic_id'
target_col = 'target'
group_col = 'patient_id'

num_cols = [
        'age_approx',                        
        'clin_size_long_diam_mm',            
        'tbp_lv_A',                          
        'tbp_lv_Aext',                       
        'tbp_lv_B',                          
        'tbp_lv_Bext',                       
        'tbp_lv_C',                          
        'tbp_lv_Cext',                       
        'tbp_lv_H',                          
        'tbp_lv_Hext',                       
        'tbp_lv_L',                          
        'tbp_lv_Lext',                       
        'tbp_lv_areaMM2',                    
        'tbp_lv_area_perim_ratio',  
        'tbp_lv_color_std_mean',         
        'tbp_lv_deltaA',                     
        'tbp_lv_deltaB',                     
        'tbp_lv_deltaL',                     
        'tbp_lv_deltaLB',                    
        'tbp_lv_deltaLBnorm', 
        'tbp_lv_eccentricity',               
        'tbp_lv_minorAxisMM',                
        'tbp_lv_nevi_confidence',            
        'tbp_lv_norm_border',                
        'tbp_lv_norm_color',                 
        'tbp_lv_perimeterMM',                
        'tbp_lv_radial_color_std_max',       
        'tbp_lv_stdL',                       
        'tbp_lv_stdLExt',                    
        'tbp_lv_symm_2axis',                 
        'tbp_lv_symm_2axis_angle',           
        'tbp_lv_x',                          
        'tbp_lv_y',                          
        'tbp_lv_z',                          
]

new_num_cols = [
        'lesion_size_ratio',             
        'lesion_shape_index',            
        'hue_contrast',                  
        'luminance_contrast',            
        'lesion_color_difference',         
        'border_complexity',           
        'color_uniformity',              

        'position_distance_3d',          
        'perimeter_to_area_ratio',       
        'area_to_perimeter_ratio',   #    
        'lesion_visibility_score',       
        'symmetry_border_consistency',   
        'consistency_symmetry_border',   #

        'color_consistency',             
        'consistency_color',           #  
        'size_age_interaction',          
        'hue_color_std_interaction',     
        'lesion_severity_index',        
        'shape_complexity_index',        #
        'color_contrast_index',          

        'log_lesion_area',               
        'normalized_lesion_size',        
        'mean_hue_difference',           
        'std_dev_contrast',              
        'color_shape_composite_index',   
        'lesion_orientation_3d',         
        'overall_color_difference',      

        'symmetry_perimeter_interaction',
        'comprehensive_lesion_index',    
        'color_variance_ratio',          #
        'border_color_interaction',      
        'border_color_interaction_2',
        'size_color_contrast_ratio',     
        'age_normalized_nevi_confidence',
        'age_normalized_nevi_confidence_2',
        'color_asymmetry_index',         

        'volume_approximation_3d',       
        'color_range',                   
        'shape_color_consistency',       
        'border_length_ratio',           
        'age_size_symmetry_index',       
        'index_age_size_symmetry',       
]
    
cat_cols = ['sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location', 'tbp_lv_location_simple', 'attribution']
norm_cols = [f'{col}_patient_norm' for col in num_cols + new_num_cols]
special_cols = ['count_per_patient', "tbp_lv_areaMM2_patient", "tbp_lv_areaMM2_bp"]
feature_cols = num_cols + new_num_cols + cat_cols + norm_cols + special_cols
def first_batch(df) :
    
    lesion_size_ratio = (pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm")).alias("lesion_size_ratio")
    lesion_shape_index = (pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2)).alias("lesion_shape_index")
    hue_contrast = (pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs().alias("hue_contrast")
    luminance_contrast = (pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs().alias("luminance_contrast")
    lesion_color_difference = ((pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2).sqrt()).alias("lesion_color_difference")
    border_complexity = (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis")).alias("border_complexity")
    color_uniformity = (pl.col("tbp_lv_color_std_mean") / (pl.col("tbp_lv_radial_color_std_max") + err)).alias("color_uniformity")

    df = df.with_columns([lesion_size_ratio, lesion_shape_index, hue_contrast, luminance_contrast,lesion_color_difference, border_complexity, color_uniformity])
    print("Total columns created in first batch - 7")
    return df
def second_batch(df) :
    
    position_distance_3d = ((pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt()).alias("position_distance_3d")
    perimeter_to_area_ratio = (pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2")).alias("perimeter_to_area_ratio")
    lesion_visibility_score = (pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color")).alias("lesion_visibility_score")
    symmetry_border_consistency = (pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border")).alias("symmetry_border_consistency")
    consistency_symmetry_border = (pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border") / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border") + err)).alias("consistency_symmetry_border")
    color_consistency = (pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext")).alias("color_consistency")
    consistency_color = (pl.col("tbp_lv_stdL") * pl.col("tbp_lv_Lext") / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext") + err)).alias("consistency_color")

    df = df.with_columns([
        position_distance_3d, perimeter_to_area_ratio, lesion_visibility_score, symmetry_border_consistency, consistency_symmetry_border, color_consistency, consistency_color
    ])
    print("Total columns created in second batch - 7")
    return df
def third_batch(df) :

    size_age_interaction = (pl.col("clin_size_long_diam_mm") * pl.col("age_approx")).alias("size_age_interaction")
    hue_color_std_interaction = (pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean")).alias("hue_color_std_interaction")
    lesion_severity_index = ((pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")) / 3).alias("lesion_severity_index")
    shape_complexity_index = (pl.col("border_complexity") + pl.col("lesion_shape_index")).alias("shape_complexity_index")
    color_contrast_index = (pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL") + pl.col("tbp_lv_deltaLBnorm")).alias("color_contrast_index")
    symmetry_perimeter_interaction = (pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM")).alias("symmetry_perimeter_interaction")
    comprehensive_lesion_index = ((pl.col("tbp_lv_area_perim_ratio") + pl.col("tbp_lv_eccentricity") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_symm_2axis")) / 4).alias("comprehensive_lesion_index")

    df = df.with_columns([
        size_age_interaction, hue_color_std_interaction, lesion_severity_index, shape_complexity_index, color_contrast_index, symmetry_perimeter_interaction, comprehensive_lesion_index
    ])
    print("Total columns created in third batch - 7")
    return df
def fourth_batch(df) :

    log_lesion_area = (pl.col("tbp_lv_areaMM2") + 1).log().alias("log_lesion_area")
    normalized_lesion_size = (pl.col("clin_size_long_diam_mm") / (pl.col("age_approx") + err)).alias("normalized_lesion_size")
    mean_hue_difference = ((pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2).alias("mean_hue_difference")
    std_dev_contrast = (((pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2) / 3).sqrt()).alias("std_dev_contrast")
    color_shape_composite_index = ((pl.col("tbp_lv_color_std_mean") + pl.col("tbp_lv_area_perim_ratio") + pl.col("tbp_lv_symm_2axis")) / 3).alias("color_shape_composite_index")
    lesion_orientation_3d = pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x")).alias("lesion_orientation_3d")
    area_to_perimeter_ratio = (pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM")).alias("area_to_perimeter_ratio")

    df = df.with_columns([
        log_lesion_area, normalized_lesion_size, mean_hue_difference, std_dev_contrast, color_shape_composite_index, lesion_orientation_3d, area_to_perimeter_ratio
    ])
    print("Total columns created in fourth batch - 7")
    return df
def fifth_batch(df) :

    overall_color_difference = ((pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")) / 3).alias("overall_color_difference")
    color_variance_ratio = (pl.col("tbp_lv_color_std_mean") / (pl.col("tbp_lv_stdLExt") + err)).alias("color_variance_ratio")
    border_color_interaction = (pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color")).alias("border_color_interaction")
    border_color_interaction_2 = (pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color") / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + err)).alias("border_color_interaction_2")
    size_color_contrast_ratio = (pl.col("clin_size_long_diam_mm") / (pl.col("tbp_lv_deltaLBnorm") + err)).alias("size_color_contrast_ratio")
    age_normalized_nevi_confidence = (pl.col("tbp_lv_nevi_confidence") / (pl.col("age_approx") + err)).alias("age_normalized_nevi_confidence")
    age_normalized_nevi_confidence_2 = ((pl.col("clin_size_long_diam_mm")**2 + pl.col("age_approx")**2).sqrt()).alias("age_normalized_nevi_confidence_2")

    df = df.with_columns([
        overall_color_difference, color_variance_ratio, border_color_interaction, border_color_interaction_2, size_color_contrast_ratio, age_normalized_nevi_confidence, age_normalized_nevi_confidence_2
    ])
    print("Total columns created in fifth batch - 7")
    return df
def sixth_batch(df) :

    color_asymmetry_index = (pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis")).alias("color_asymmetry_index")
    volume_approximation_3d = (pl.col("tbp_lv_areaMM2") * (pl.col("tbp_lv_x")**2 + pl.col("tbp_lv_y")**2 + pl.col("tbp_lv_z")**2).sqrt()).alias("volume_approximation_3d")
    color_range = ((pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs() + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs() + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs()).alias("color_range")
    shape_color_consistency = (pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean")).alias("shape_color_consistency")
    border_length_ratio = (pl.col("tbp_lv_perimeterMM") / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt())).alias("border_length_ratio")
    age_size_symmetry_index = (pl.col("age_approx") * pl.col("clin_size_long_diam_mm") * pl.col("tbp_lv_symm_2axis")).alias("age_size_symmetry_index")
    index_age_size_symmetry = (pl.col("age_approx") * pl.col("tbp_lv_areaMM2") * pl.col("tbp_lv_symm_2axis")).alias("index_age_size_symmetry")

    df = df.with_columns([
        color_asymmetry_index, volume_approximation_3d, color_range, shape_color_consistency, border_length_ratio, age_size_symmetry_index, index_age_size_symmetry
    ])
    print("Total columns created in sixth batch - 7")
    return df
def read_data(path) :
    # 1) load
    df = pl.read_csv(path)
    err = 1e-5
    id_col = 'isic_id'
    target_col = 'target'
    group_col = 'patient_id'
    num_cols = [
        'age_approx',                        
        'clin_size_long_diam_mm',            
        'tbp_lv_A',                          
        'tbp_lv_Aext',                       
        'tbp_lv_B',                          
        'tbp_lv_Bext',                       
        'tbp_lv_C',                          
        'tbp_lv_Cext',                       
        'tbp_lv_H',                          
        'tbp_lv_Hext',                       
        'tbp_lv_L',                          
        'tbp_lv_Lext',                       
        'tbp_lv_areaMM2',                    
        'tbp_lv_area_perim_ratio',  
        'tbp_lv_color_std_mean',         
        'tbp_lv_deltaA',                     
        'tbp_lv_deltaB',                     
        'tbp_lv_deltaL',                     
        'tbp_lv_deltaLB',                    
        'tbp_lv_deltaLBnorm', 
        'tbp_lv_eccentricity',               
        'tbp_lv_minorAxisMM',                
        'tbp_lv_nevi_confidence',            
        'tbp_lv_norm_border',                
        'tbp_lv_norm_color',                 
        'tbp_lv_perimeterMM',                
        'tbp_lv_radial_color_std_max',       
        'tbp_lv_stdL',                       
        'tbp_lv_stdLExt',                    
        'tbp_lv_symm_2axis',                 
        'tbp_lv_symm_2axis_angle',           
        'tbp_lv_x',                          
        'tbp_lv_y',                          
        'tbp_lv_z',                          
    ]

    new_num_cols = [
        'lesion_size_ratio',             
        'lesion_shape_index',            
        'hue_contrast',                  
        'luminance_contrast',            
        'lesion_color_difference',         
        'border_complexity',           
        'color_uniformity',              

        'position_distance_3d',          
        'perimeter_to_area_ratio',       
        'area_to_perimeter_ratio',       
        'lesion_visibility_score',       
        'symmetry_border_consistency',   
        'consistency_symmetry_border',   

        'color_consistency',             
        'consistency_color',             
        'size_age_interaction',          
        'hue_color_std_interaction',     
        'lesion_severity_index',        
        'shape_complexity_index',        
        'color_contrast_index',          

        'log_lesion_area',               
        'normalized_lesion_size',        
        'mean_hue_difference',           
        'std_dev_contrast',              
        'color_shape_composite_index',   
        'lesion_orientation_3d',         
        'overall_color_difference',      

        'symmetry_perimeter_interaction',
        'comprehensive_lesion_index',    
        'color_variance_ratio',          
        'border_color_interaction',      
        'border_color_interaction_2',
        'size_color_contrast_ratio',     
        'age_normalized_nevi_confidence',
        'age_normalized_nevi_confidence_2',
        'color_asymmetry_index',         

        'volume_approximation_3d',       
        'color_range',                   
        'shape_color_consistency',       
        'border_length_ratio',           
        'age_size_symmetry_index',       
        'index_age_size_symmetry',       
    ]
    
    cat_cols = ['sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location', 'tbp_lv_location_simple', 'attribution']
    
    # 2) clean age_approx
    age_clean = (
        pl.col("age_approx")
        .cast(pl.String)
        .replace("NA", np.nan)
        .cast(pl.Float64)
        .alias("age_approx")
    )
    df = df.with_columns(age_clean)

    # 3) impute all float columns with median
    float_impute = pl.col(pl.Float64).fill_nan(pl.col(pl.Float64).median())
    df = df.with_columns(float_impute)

    # 4) engineered features (create then add)
    df = first_batch(df)
    df = second_batch(df)
    df = third_batch(df)
    df = fourth_batch(df)
    df = fifth_batch(df)
    df = sixth_batch(df)

    combined_anatomical_site = (pl.col("anatom_site_general") + pl.lit("_") + pl.col("tbp_lv_location")).alias("combined_anatomical_site")
    df = df.with_columns([combined_anatomical_site])
    
    # 5) per-patient z-norm for numeric columns
    patient_norm_exprs = [
        ((pl.col(c) - pl.col(c).mean().over("patient_id")) / (pl.col(c).std().over("patient_id") + err)).alias(f"{c}_patient_norm")
        for c in (num_cols + new_num_cols)
    ]
    df = df.with_columns(patient_norm_exprs)

    df = df.with_columns(pl.col('tbp_lv_areaMM2').sum().over('patient_id').alias("tbp_lv_areaMM2_patient"))
    df = df.with_columns(pl.col('tbp_lv_areaMM2').sum().over(['patient_id', 'anatom_site_general']).alias("tbp_lv_areaMM2_bp"))
    df = df.with_columns(pl.col("isic_id").count().over("patient_id").alias("count_per_patient"))

    # 8) cast categoricals
    df = df.with_columns(pl.col(cat_cols).cast(pl.Categorical))

    # 9) return pandas with index
    return df.to_pandas().set_index(id_col)
def preprocess(df_train, df_test, feature_cols, cat_cols):
    
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
    encoder.fit(df_train[cat_cols])
    
    new_cat_cols = [f'onehot_{i}' for i in range(len(encoder.get_feature_names_out()))]

    df_train[new_cat_cols] = encoder.transform(df_train[cat_cols])
    df_train[new_cat_cols] = df_train[new_cat_cols].astype('category')

    df_test[new_cat_cols] = encoder.transform(df_test[cat_cols])
    df_test[new_cat_cols] = df_test[new_cat_cols].astype('category')

    for col in cat_cols:
        feature_cols.remove(col)

    feature_cols.extend(new_cat_cols)
    cat_cols = new_cat_cols
    
    return df_train, df_test, feature_cols, cat_cols
df_train = read_data(train_path)
print('train data loaded and feature engineering complete')
df_test = read_data(test_path)
print('test data loaded and feature engineering complete')

df_train, df_test, feature_cols, cat_cols  = preprocess(df_train, df_test, feature_cols, cat_cols)
print('train and test data preprocessing complete')
print(f"num_cols       : {len(num_cols)}")
print(f"new_num_cols   : {len(new_num_cols)}")
print(f"cat_cols       : {len(cat_cols)}")
print(f"special_cols   : {len(special_cols)}")
print(f"norm_cols      : {len(norm_cols)}")

# Print total
print(f"TOTAL features : {len(feature_cols)}")
def custom_metric_raw(y_hat, y_true):
    y_true = np.array(y_true)   # ensure numpy array
    y_hat = np.array(y_hat)     # ensure numpy array
    
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)
    
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc


def custom_metric(estimator, X, y_true):
    y_hat = estimator.predict_proba(X)[:, 1]
    partial_auc = custom_metric_raw(y_hat, y_true)
    return partial_auc
    
lgb_params = {
    'objective':        'binary',
    'verbosity':        -1,
    'n_iter':           200,
    'boosting_type':    'gbdt',
    "device" : "gpu",
    'random_state':     seed,
    'lambda_l1':        0.08758718919397321, 
    'lambda_l2':        0.0039689175176025465, 
    'learning_rate':    0.03231007103195577, 
    'max_depth':        4, 
    'num_leaves':       103, 
    'colsample_bytree': 0.8329551585827726, 
    'colsample_bynode': 0.4025961355653304, 
    'bagging_fraction': 0.7738954452473223, 
    'bagging_freq':     4, 
    'min_data_in_leaf': 85, 
    'scale_pos_weight': 2.7984184778875543,
}

cb_params = {
    'loss_function':     'Logloss',
    'iterations':        200,
    'verbose':           False,
    'random_state':      seed,
    'max_depth':         7, 
    'learning_rate':     0.06936242010150652, 
    'scale_pos_weight':  2.6149345838209532, 
    'l2_leaf_reg':       6.216113851699493, 
    'subsample':         0.6249261779711819, 
    'min_data_in_leaf':  24,
    'cat_features':      cat_cols,
}

xgb_params  = {
    'enable_categorical': True,
    'tree_method':        'hist',
    'random_state':       seed,
    'learning_rate':      0.08501257473292347, 
    'lambda':             8.879624125465703, 
    'alpha':              0.6779926606782505, 
    'max_depth':          6, 
    'subsample':          0.6012681388711075, 
    'colsample_bytree':   0.8437772277074493, 
    'colsample_bylevel':  0.5476090898823716, 
    'colsample_bynode':   0.9928601203635129, 
    'scale_pos_weight':   3.29440313334688,
    "device": "cuda"
}

def run_model_lgb(lgb_params, reduce=True, columns_to_drop=None) -> float:
    columns_to_drop = [] if columns_to_drop is None else columns_to_drop
    metric_list = []
    models = []
    for random_seed in range(1, 10):
        random_seed = random_seed * 10 + 17
        tsp = StratifiedGroupKFold(5, shuffle=True, random_state=random_seed)
        metrics_ev_df = []
        test_forecast = []
        val_forecast = []
        for fold_n, (train_index, val_index) in enumerate(tsp.split(df_train, y=df_train.target, groups=df_train[group_col])):
            train_slice_x = df_train.iloc[train_index][[i for i in feature_cols if i not in columns_to_drop]].reset_index(drop=True)
            val_slice_x = df_train.iloc[val_index][[i for i in feature_cols if i not in columns_to_drop]].reset_index(drop=True)
                
            train_slice_y = df_train.iloc[train_index]['target'].reset_index(drop=True)
            val_slice_y = df_train.iloc[val_index]['target'].reset_index(drop=True)
        
            cb_model = Pipeline([
                ('over', RandomOverSampler(sampling_strategy={1: 2000})),  # boost positives
                ('under', RandomUnderSampler(sampling_strategy={0: 10000})),  # reduce negatives
                ('classifier', lgb.LGBMClassifier(**lgb_params))
            ])
            
            cb_model.fit(train_slice_x, train_slice_y)
            preds_cb = cb_model.predict_proba(val_slice_x)[:, 1]
            metric = custom_metric_raw(preds_cb, val_slice_y.values)
            metric_list.append(metric)
            models.append(cb_model)

    if reduce:
        return np.mean(metric_list), models
    else:
        return metric_list, models
    
metric_list, models_xgb = run_model_lgb(
    xgb_params, reduce=False, columns_to_drop=None)

print(np.mean(metric_list))