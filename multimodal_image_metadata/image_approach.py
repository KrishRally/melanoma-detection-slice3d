import os
import numpy as np
import pandas as pd
import polars as pl
import cv2
import itertools
import h5py
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torchvision.models as models
from torchvision import transforms
import timm
from PIL import Image
class SLICE3DDataExtractor:
    """Extract and organize SLICE-3D dataset components - CORRECTED VERSION"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.train_csv = None
        self.test_csv = None
        self.image_paths = {}
        
    def extract_data(self):
        """Extract all dataset components"""
        print("🔄 Extracting SLICE-3D dataset...")
        
        # Load CSV files
        try:
            train_metadata_path = os.path.join(self.data_path, 'train-metadata.csv')
            test_metadata_path = os.path.join(self.data_path, 'test-metadata.csv')
            
            if not os.path.exists(train_metadata_path):
                raise FileNotFoundError(f"Training metadata not found: {train_metadata_path}")
            if not os.path.exists(test_metadata_path):
                raise FileNotFoundError(f"Test metadata not found: {test_metadata_path}")
                
            self.train_csv = pd.read_csv(train_metadata_path)
            self.test_csv = pd.read_csv(test_metadata_path)
                
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            raise
        
        # CORRECTED: Proper path construction for nested folders
        train_img_dir = os.path.join(self.data_path, 'train-image', 'processed_image')
        
        # Handle test images - check if test-image exists or if it's in HDF5 format
        test_img_dir = None
        if os.path.exists(os.path.join(self.data_path, 'test-image')):
            test_img_dir = os.path.join(self.data_path, 'test-image', 'image')
        
        # Verify directories exist
        if not os.path.exists(train_img_dir):
            raise FileNotFoundError(f"Training image directory not found: {train_img_dir}")
        
        print(f"Training image directory: {train_img_dir}")
        print(f"Test image directory: {test_img_dir}")
        
        # Map training image paths
        missing_train_images = []
        for idx, row in self.train_csv.iterrows():
            image_path = os.path.join(train_img_dir, f"{row['isic_id']}.jpg")
            if os.path.exists(image_path):
                self.image_paths[row['isic_id']] = image_path
            else:
                missing_train_images.append(row['isic_id'])
        
        if missing_train_images:
            print(f"Warning: {len(missing_train_images)} training images not found")
            print(f"First few missing: {missing_train_images[:5]}")
        
        # Map test image paths (if directory exists)
        missing_test_images = []
        test_img_dir = None

        # First check for HDF5 format
        if self.handle_test_images_hdf5():
            print("Test images are in HDF5 format - will handle during prediction")
        else:
            # Your original test image handling code works fine
            if os.path.exists(os.path.join(self.data_path, 'test-image')):
                test_img_dir = os.path.join(self.data_path, 'test-image', 'image')
    
            if test_img_dir and os.path.exists(test_img_dir):
                for idx, row in self.test_csv.iterrows():
                    image_path = os.path.join(test_img_dir, f"{row['isic_id']}.jpg")
                    if os.path.exists(image_path):
                        self.image_paths[row['isic_id']] = image_path
                    else:
                        missing_test_images.append(row['isic_id'])
                
                if missing_test_images:
                    print(f"Warning: {len(missing_test_images)} test images not found")
            else:
                print("Test images not found in directory structure - may be in HDF5 format")
    
    def extract_hdf5_images(self, hdf5_file, output_dir):
        """Helper function to extract images from HDF5 files if needed"""
        import h5py
        
        print(f"Extracting images from {hdf5_file}...")
        os.makedirs(output_dir, exist_ok=True)
        
        with h5py.File(os.path.join(self.data_path, hdf5_file), 'r') as f:
            # Inspect HDF5 structure
            print("HDF5 structure:")
            def print_structure(name, obj):
                print(f"  {name}: {type(obj)}")
            f.visititems(print_structure)
            
            # Extract images (structure may vary)
            # This is a template - adjust based on actual HDF5 structure
            if 'images' in f:
                images = f['images']
                for i, img_data in enumerate(images):
                    # Save image (adjust format as needed)
                    img_path = os.path.join(output_dir, f"image_{i}.jpg")
                    # Convert and save logic here
                    pass
                    
    def handle_test_images_hdf5(self):
        """Handle test images if they're in HDF5 format"""
        test_hdf5_path = os.path.join(self.data_path, 'test-image.hdf5')
    
        if os.path.exists(test_hdf5_path):
            print(f"Test images found in HDF5 format: {test_hdf5_path}")
        
            # For training phase, we can skip test image extraction
            print("Skipping test image extraction for now (training phase)")
            print("Test images will be handled during prediction phase")
            return True
    
        return False
class ImagePreprocessor:
    """Advanced preprocessing while preserving native image dimensions"""
    
    def __init__(self, max_size=1024, min_size=224):
        self.max_size = max_size
        self.min_size = min_size
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # Converts to tensor and normalizes to [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

    def smart_resize(self, image):
        """Intelligent resizing that preserves aspect ratio and doesn't force specific dimensions"""
        h, w = image.shape[:2]
        
        # Only resize if image is too large or too small
        if max(h, w) > self.max_size:
            # Scale down large images
            scale = self.max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        elif max(h, w) < self.min_size:
            # Scale up very small images
            scale = self.min_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
        return image
    
    def remove_hair_artifacts(self, image):
        """Advanced hair removal using morphological operations and inpainting"""
        # Convert to grayscale for hair detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        
        # Apply black-hat morphological filter
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Create hair mask
        _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Apply Gaussian blur to smooth the mask
        hair_mask = cv2.GaussianBlur(hair_mask, (3, 3), 0)
        
        # Inpaint hair regions
        result = cv2.inpaint(image, hair_mask, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def color_normalization(self, image):
        """Advanced color normalization for cross-domain generalization"""
        # Convert to LAB color space for better color handling
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def process_image(self, image_path):
        """Complete image preprocessing pipeline"""
        # Load image
       # image = cv2.imread(image_path)
       # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
       # Apply preprocessing steps
       # image = self.smart_resize(image)
      #  image = self.remove_hair_artifacts(image)
       # image = self.color_normalization(image)
       # image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        from PIL import Image
        pil_image = Image.open(image_path)
            
        # Apply transforms to get proper tensor
        tensor_image = self.transforms(pil_image)
            
        return tensor_image
class FeatureEngineer:
    """Comprehensive feature engineering for metadata"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
        self.feature_names = []
        self.fitted = False
        
        # Configuration
        self.err = 1e-5
        self.id_col = 'isic_id'
        self.target_col = 'target'
        self.group_col = 'patient_id'
        
        # Column definitions
        self.num_cols = [
            'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext',
            'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H',
            'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2',
            'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA',
            'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm',
            'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence',
            'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM',
            'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt',
            'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z'
        ]
        
        self.new_num_cols = [
            'lesion_size_ratio', 'lesion_shape_index', 'hue_contrast', 'luminance_contrast',
            'lesion_color_difference', 'border_complexity', 'color_uniformity',
            'position_distance_3d', 'perimeter_to_area_ratio', 'area_to_perimeter_ratio',
            'lesion_visibility_score', 'symmetry_border_consistency', 'consistency_symmetry_border',
            'color_consistency', 'consistency_color', 'size_age_interaction',
            'hue_color_std_interaction', 'lesion_severity_index', 'shape_complexity_index',
            'color_contrast_index', 'log_lesion_area', 'normalized_lesion_size',
            'mean_hue_difference', 'std_dev_contrast', 'color_shape_composite_index',
            'lesion_orientation_3d', 'overall_color_difference', 'symmetry_perimeter_interaction',
            'comprehensive_lesion_index', 'color_variance_ratio', 'border_color_interaction',
            'border_color_interaction_2', 'size_color_contrast_ratio', 'age_normalized_nevi_confidence',
            'age_normalized_nevi_confidence_2', 'color_asymmetry_index', 'volume_approximation_3d',
            'color_range', 'shape_color_consistency', 'border_length_ratio',
            'age_size_symmetry_index', 'index_age_size_symmetry'
        ]
        
        self.cat_cols = ['sex', 'anatom_site_general', 'tbp_lv_location', 
                        'tbp_lv_location_simple']
        
        self.norm_cols = [f'{col}_patient_norm' for col in self.num_cols + self.new_num_cols]
        self.special_cols = ['count_per_patient', "tbp_lv_areaMM2_patient", "tbp_lv_areaMM2_bp"]
        self.feature_cols = self.num_cols + self.new_num_cols + self.cat_cols + self.norm_cols + self.special_cols

    def _first_batch(self, df):
        """Create first batch of engineered features"""
        lesion_size_ratio = (pl.col("tbp_lv_minorAxisMM") / pl.col("clin_size_long_diam_mm")).alias("lesion_size_ratio")
        lesion_shape_index = (pl.col("tbp_lv_areaMM2") / (pl.col("tbp_lv_perimeterMM") ** 2)).alias("lesion_shape_index")
        hue_contrast = (pl.col("tbp_lv_H") - pl.col("tbp_lv_Hext")).abs().alias("hue_contrast")
        luminance_contrast = (pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs().alias("luminance_contrast")
        lesion_color_difference = ((pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2).sqrt()).alias("lesion_color_difference")
        border_complexity = (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_symm_2axis")).alias("border_complexity")
        color_uniformity = (pl.col("tbp_lv_color_std_mean") / (pl.col("tbp_lv_radial_color_std_max") + self.err)).alias("color_uniformity")

        df = df.with_columns([lesion_size_ratio, lesion_shape_index, hue_contrast, luminance_contrast,
                             lesion_color_difference, border_complexity, color_uniformity])
        return df

    def _second_batch(self, df):
        """Create second batch of engineered features"""
        position_distance_3d = ((pl.col("tbp_lv_x") ** 2 + pl.col("tbp_lv_y") ** 2 + pl.col("tbp_lv_z") ** 2).sqrt()).alias("position_distance_3d")
        perimeter_to_area_ratio = (pl.col("tbp_lv_perimeterMM") / pl.col("tbp_lv_areaMM2")).alias("perimeter_to_area_ratio")
        lesion_visibility_score = (pl.col("tbp_lv_deltaLBnorm") + pl.col("tbp_lv_norm_color")).alias("lesion_visibility_score")
        symmetry_border_consistency = (pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border")).alias("symmetry_border_consistency")
        consistency_symmetry_border = (pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_norm_border") / (pl.col("tbp_lv_symm_2axis") + pl.col("tbp_lv_norm_border") + self.err)).alias("consistency_symmetry_border")
        color_consistency = (pl.col("tbp_lv_stdL") / pl.col("tbp_lv_Lext")).alias("color_consistency")
        consistency_color = (pl.col("tbp_lv_stdL") * pl.col("tbp_lv_Lext") / (pl.col("tbp_lv_stdL") + pl.col("tbp_lv_Lext") + self.err)).alias("consistency_color")

        df = df.with_columns([position_distance_3d, perimeter_to_area_ratio, lesion_visibility_score,
                             symmetry_border_consistency, consistency_symmetry_border, color_consistency, consistency_color])
        return df

    def _third_batch(self, df):
        """Create third batch of engineered features"""
        size_age_interaction = (pl.col("clin_size_long_diam_mm") * pl.col("age_approx")).alias("size_age_interaction")
        hue_color_std_interaction = (pl.col("tbp_lv_H") * pl.col("tbp_lv_color_std_mean")).alias("hue_color_std_interaction")
        lesion_severity_index = ((pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_eccentricity")) / 3).alias("lesion_severity_index")
        shape_complexity_index = (pl.col("border_complexity") + pl.col("lesion_shape_index")).alias("shape_complexity_index")
        color_contrast_index = (pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL") + pl.col("tbp_lv_deltaLBnorm")).alias("color_contrast_index")
        symmetry_perimeter_interaction = (pl.col("tbp_lv_symm_2axis") * pl.col("tbp_lv_perimeterMM")).alias("symmetry_perimeter_interaction")
        comprehensive_lesion_index = ((pl.col("tbp_lv_area_perim_ratio") + pl.col("tbp_lv_eccentricity") + pl.col("tbp_lv_norm_color") + pl.col("tbp_lv_symm_2axis")) / 4).alias("comprehensive_lesion_index")

        df = df.with_columns([size_age_interaction, hue_color_std_interaction, lesion_severity_index,
                             shape_complexity_index, color_contrast_index, symmetry_perimeter_interaction, comprehensive_lesion_index])
        return df

    def _fourth_batch(self, df):
        """Create fourth batch of engineered features"""
        log_lesion_area = (pl.col("tbp_lv_areaMM2") + 1).log().alias("log_lesion_area")
        normalized_lesion_size = (pl.col("clin_size_long_diam_mm") / (pl.col("age_approx") + self.err)).alias("normalized_lesion_size")
        mean_hue_difference = ((pl.col("tbp_lv_H") + pl.col("tbp_lv_Hext")) / 2).alias("mean_hue_difference")
        std_dev_contrast = (((pl.col("tbp_lv_deltaA") ** 2 + pl.col("tbp_lv_deltaB") ** 2 + pl.col("tbp_lv_deltaL") ** 2) / 3).sqrt()).alias("std_dev_contrast")
        color_shape_composite_index = ((pl.col("tbp_lv_color_std_mean") + pl.col("tbp_lv_area_perim_ratio") + pl.col("tbp_lv_symm_2axis")) / 3).alias("color_shape_composite_index")
        lesion_orientation_3d = pl.arctan2(pl.col("tbp_lv_y"), pl.col("tbp_lv_x")).alias("lesion_orientation_3d")
        area_to_perimeter_ratio = (pl.col("tbp_lv_areaMM2") / pl.col("tbp_lv_perimeterMM")).alias("area_to_perimeter_ratio")

        df = df.with_columns([log_lesion_area, normalized_lesion_size, mean_hue_difference, std_dev_contrast,
                             color_shape_composite_index, lesion_orientation_3d, area_to_perimeter_ratio])
        return df

    def _fifth_batch(self, df):
        """Create fifth batch of engineered features"""
        overall_color_difference = ((pl.col("tbp_lv_deltaA") + pl.col("tbp_lv_deltaB") + pl.col("tbp_lv_deltaL")) / 3).alias("overall_color_difference")
        color_variance_ratio = (pl.col("tbp_lv_color_std_mean") / (pl.col("tbp_lv_stdLExt") + self.err)).alias("color_variance_ratio")
        border_color_interaction = (pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color")).alias("border_color_interaction")
        border_color_interaction_2 = (pl.col("tbp_lv_norm_border") * pl.col("tbp_lv_norm_color") / (pl.col("tbp_lv_norm_border") + pl.col("tbp_lv_norm_color") + self.err)).alias("border_color_interaction_2")
        size_color_contrast_ratio = (pl.col("clin_size_long_diam_mm") / (pl.col("tbp_lv_deltaLBnorm") + self.err)).alias("size_color_contrast_ratio")
        age_normalized_nevi_confidence = (pl.col("tbp_lv_nevi_confidence") / (pl.col("age_approx") + self.err)).alias("age_normalized_nevi_confidence")
        age_normalized_nevi_confidence_2 = ((pl.col("clin_size_long_diam_mm")**2 + pl.col("age_approx")**2).sqrt()).alias("age_normalized_nevi_confidence_2")

        df = df.with_columns([overall_color_difference, color_variance_ratio, border_color_interaction,
                             border_color_interaction_2, size_color_contrast_ratio, age_normalized_nevi_confidence, age_normalized_nevi_confidence_2])
        return df

    def _sixth_batch(self, df):
        """Create sixth batch of engineered features"""
        color_asymmetry_index = (pl.col("tbp_lv_radial_color_std_max") * pl.col("tbp_lv_symm_2axis")).alias("color_asymmetry_index")
        volume_approximation_3d = (pl.col("tbp_lv_areaMM2") * (pl.col("tbp_lv_x")**2 + pl.col("tbp_lv_y")**2 + pl.col("tbp_lv_z")**2).sqrt()).alias("volume_approximation_3d")
        color_range = ((pl.col("tbp_lv_L") - pl.col("tbp_lv_Lext")).abs() + (pl.col("tbp_lv_A") - pl.col("tbp_lv_Aext")).abs() + (pl.col("tbp_lv_B") - pl.col("tbp_lv_Bext")).abs()).alias("color_range")
        shape_color_consistency = (pl.col("tbp_lv_eccentricity") * pl.col("tbp_lv_color_std_mean")).alias("shape_color_consistency")
        border_length_ratio = (pl.col("tbp_lv_perimeterMM") / (2 * np.pi * (pl.col("tbp_lv_areaMM2") / np.pi).sqrt())).alias("border_length_ratio")
        age_size_symmetry_index = (pl.col("age_approx") * pl.col("clin_size_long_diam_mm") * pl.col("tbp_lv_symm_2axis")).alias("age_size_symmetry_index")
        index_age_size_symmetry = (pl.col("age_approx") * pl.col("tbp_lv_areaMM2") * pl.col("tbp_lv_symm_2axis")).alias("index_age_size_symmetry")

        df = df.with_columns([color_asymmetry_index, volume_approximation_3d, color_range, shape_color_consistency,
                             border_length_ratio, age_size_symmetry_index, index_age_size_symmetry])
        return df

    def _process_dataframe(self, df):
        """Apply feature engineering to a pandas DataFrame"""
        # Convert pandas to polars for processing
        if isinstance(df, pd.DataFrame):
            # Reset index if it's set to preserve the ID column
            if df.index.name == self.id_col:
                df_reset = df.reset_index()
            else:
                df_reset = df.copy()
            df_pl = pl.from_pandas(df_reset)
        else:
            df_pl = df
        
        # Clean age_approx
        age_clean = (
            pl.col("age_approx")
            .cast(pl.String)
            .replace("NA", np.nan)
            .cast(pl.Float64)
            .alias("age_approx")
        )
        df_pl = df_pl.with_columns(age_clean)

        # Impute all float columns with median
        float_impute = pl.col(pl.Float64).fill_nan(pl.col(pl.Float64).median())
        df_pl = df_pl.with_columns(float_impute)

        # Apply feature engineering batches
        df_pl = self._first_batch(df_pl)
        df_pl = self._second_batch(df_pl)
        df_pl = self._third_batch(df_pl)
        df_pl = self._fourth_batch(df_pl)
        df_pl = self._fifth_batch(df_pl)
        df_pl = self._sixth_batch(df_pl)

        # Create combined anatomical site
        combined_anatomical_site = (pl.col("anatom_site_general") + pl.lit("_") + pl.col("tbp_lv_location")).alias("combined_anatomical_site")
        df_pl = df_pl.with_columns([combined_anatomical_site])
        
        # Per-patient z-normalization for numeric columns
        patient_norm_exprs = [
            ((pl.col(c) - pl.col(c).mean().over("patient_id")) / (pl.col(c).std().over("patient_id") + self.err)).alias(f"{c}_patient_norm")
            for c in (self.num_cols + self.new_num_cols)
        ]
        df_pl = df_pl.with_columns(patient_norm_exprs)

        # Additional patient-level features
        df_pl = df_pl.with_columns(pl.col('tbp_lv_areaMM2').sum().over('patient_id').alias("tbp_lv_areaMM2_patient"))
        df_pl = df_pl.with_columns(pl.col('tbp_lv_areaMM2').sum().over(['patient_id', 'anatom_site_general']).alias("tbp_lv_areaMM2_bp"))
        df_pl = df_pl.with_columns(pl.col("isic_id").count().over("patient_id").alias("count_per_patient"))

        # Cast categoricals
        df_pl = df_pl.with_columns(pl.col(self.cat_cols).cast(pl.Categorical))

        return df_pl.to_pandas().set_index(self.id_col)

    def create_features(self, train_df, is_training):
        """Create advanced features from metadata"""
    
        # Process the dataframe
        df_processed = self._process_dataframe(train_df)
    
        # Handle categorical encoding
        cat_cols_available = [col for col in self.cat_cols if col in df_processed.columns]
    
        if cat_cols_available:
            if is_training:
                self.encoder.fit(df_processed[cat_cols_available])
        
            # Transform categorical data
            encoded_cat_data = self.encoder.transform(df_processed[cat_cols_available])
            new_cat_cols = [f'onehot_{i}' for i in range(encoded_cat_data.shape[1])]
        
            # Add encoded categorical data as NUMERIC columns (not category)
            for i, col_name in enumerate(new_cat_cols):
                df_processed[col_name] = encoded_cat_data[:, i].astype(np.float32)
        else:
            new_cat_cols = []

        # Get final feature set (numeric + engineered + normalized + encoded categorical)
        available_features = []
    
        # Add numeric columns that exist
        for col in self.num_cols:
            if col in df_processed.columns:
                available_features.append(col)
    
        # Add engineered columns that exist  
        for col in self.new_num_cols:
            if col in df_processed.columns:
                available_features.append(col)
            
        # Add normalized columns that exist
        for col in self.norm_cols:
            if col in df_processed.columns:
                available_features.append(col)
    
        # Add encoded categorical columns (now numeric)
        available_features.extend(new_cat_cols)
    
        # Add any special columns that exist
        special_cols_available = [col for col in df_processed.columns if col.endswith('_patient') or col.endswith('_bp') or col == 'count_per_patient']
        available_features.extend(special_cols_available)
    
        # Extract feature matrix
        feature_matrix = df_processed[available_features].values
    
        # Ensure all data is float32 for PyTorch compatibility
        feature_matrix = feature_matrix.astype(np.float32)
    
        # Handle any remaining NaN or inf values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    
        # Store feature names for reference
        self.feature_names = available_features
        self.fitted = True
    
        print(f"Created {len(available_features)} total features")
        print(f"   - Original + engineered numeric features: {len([c for c in available_features if c in self.num_cols + self.new_num_cols])}")
        print(f"   - Patient-normalized features: {len([c for c in available_features if c.endswith('_patient_norm')])}")
        print(f"   - One-hot encoded categorical features: {len(new_cat_cols)}")
        print(f"   - Special features: {len(special_cols_available)}")
        print(f"   - Feature matrix shape: {feature_matrix.shape}")
        print(f"   - Feature matrix dtype: {feature_matrix.dtype}")
    
        return feature_matrix
class SLICE3DDataset(Dataset):
    """Dataset class with image and metadata fusion"""
    
    def __init__(self, df, image_paths, metadata_features,
                 preprocessor=None, is_training=True):
        self.df = df.reset_index(drop=True)
        self.image_paths = image_paths
        self.metadata_features = metadata_features
        self.preprocessor = preprocessor
        self.is_training = is_training
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        isic_id = row['isic_id']
        
        # Get image
        if isic_id in self.image_paths:
            image_path = self.image_paths[isic_id]
            
            # Apply custom preprocessing
            if self.preprocessor:
                image = self.preprocessor.process_image(image_path)
        else:
            # Create placeholder if image not found
            image = torch.zeros(3, 224, 224)
        
        # Get metadata features
        metadata = self.metadata_features[idx]
        metadata = torch.FloatTensor(metadata)

        label = row['target']

        if self.is_training:
            return image, metadata, torch.FloatTensor([label])
        else:
            return image, metadata, torch.FloatTensor([label])
def calculate_pauc(y_true, y_hat):
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)
    
    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])
    
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return partial_auc
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.6, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Calculate focal loss
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * loss
        
        return focal_loss.mean()

class Trainer:
    """Advanced training with multiple strategies"""
    
    def __init__(self, model, device, learning_rate=2e-6, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                           weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=1, eta_min=1e-7
        )
        self.focal_loss = FocalLoss()
        self.best_pauc = 0
        
    def train_epoch(self, dataloader, class_weights=None):
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch_idx, (images, metadata, labels) in enumerate(dataloader):
            images = images.to(self.device)
            metadata = metadata.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images, metadata)
            
            # Calculate loss with class weights
            if class_weights is not None:
                weights = torch.tensor([class_weights[int(l.item())] for l in labels]).to(self.device)
                weights = weights.view(-1, 1)
                loss = F.binary_cross_entropy_with_logits(outputs, labels, weight=weights)
            else:
                loss = self.focal_loss(outputs, labels)

            l2_lambda = 0.001
            l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        preds_binary = (np.array(predictions) >= 0.3).astype(int)
        targets_array = np.array(targets).astype(int)

        # Compute metrics
        pauc = calculate_pauc(targets_array, np.array(predictions))
        accuracy = accuracy_score(targets_array, preds_binary)
        precision = precision_score(targets_array, preds_binary, zero_division=0)
        recall = recall_score(targets_array, preds_binary, zero_division=0)

        return avg_loss, pauc, accuracy, precision, recall
    
    def validate(self, dataloader):
        self.model.eval()
        predictions = []
        targets = []
        total_loss = 0
        
        with torch.no_grad():
            for images, metadata, labels in dataloader:
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, metadata)

                loss = self.focal_loss(outputs, labels)
                total_loss += loss.item()
                predictions.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        preds_binary = (np.array(predictions) >= 0.3).astype(int)
        targets_array = np.array(targets).astype(int)

        # Compute metrics
        pauc = calculate_pauc(targets_array, np.array(predictions))
        accuracy = accuracy_score(targets_array, preds_binary)
        precision = precision_score(targets_array, preds_binary, zero_division=0)
        recall = recall_score(targets_array, preds_binary, zero_division=0)

        return avg_loss, pauc, accuracy, precision, recall
class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism for image and metadata"""
    
    def __init__(self, image_dim, metadata_dim, hidden_dim=256):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.metadata_proj = nn.Linear(metadata_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, image_feat, metadata_feat):
        # Project features
        img_proj = self.image_proj(image_feat).unsqueeze(1)  # [B, 1, H]
        meta_proj = self.metadata_proj(metadata_feat).unsqueeze(1)  # [B, 1, H]
        
        # Concatenate for cross-attention
        combined = torch.cat([img_proj, meta_proj], dim=1)  # [B, 2, H]
        
        # Apply attention
        attn_out, _ = self.attention(combined, combined, combined)
        attn_out = self.norm(attn_out + combined)
        
        # Aggregate
        fused = attn_out.mean(dim=1)  # [B, H]
        return fused

class SLICE3DModel(nn.Module):
    """Multi-backbone model with metadata fusion"""
    
    def __init__(self, model_name='resnet18', metadata_dim=50, dropout=0.5, 
                 fusion_strategy='attention'):
        super().__init__()
        self.model_name = model_name
        self.fusion_strategy = fusion_strategy
        
        # Initialize backbone
        if 'resnet' in model_name:
            self.backbone = models.__dict__[model_name](pretrained=True)
            for name, param in self.backbone.named_parameters():
                if 'layer4' not in name:
                    param.requires_grad = False
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif 'vgg' in model_name:
            self.backbone = models.vgg16(weights='IMAGENET1K_V1')
            # Remove classifier and get feature dimension
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Identity()
        elif 'efficientnet' in model_name:
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
            num_features = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.backbone_dropout = nn.Dropout(0.3)
        
        # Metadata processing
        self.metadata_net = nn.Sequential(
            nn.Linear(metadata_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        metadata_output_dim = self.metadata_net[-3].out_features
        
        # Fusion layer
        if fusion_strategy == 'attention':
            self.fusion = AttentionFusion(num_features, metadata_output_dim, hidden_dim=256)
            fusion_dim = 256
        elif fusion_strategy == 'concat':
            self.fusion = None
            fusion_dim = num_features + metadata_output_dim
        else:  # 'gated'
            self.gate = nn.Sequential(
                nn.Linear(num_features + metadata_output_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_features + metadata_output_dim),
                nn.Sigmoid()
            )
            self.fusion = None
            fusion_dim = num_features + metadata_output_dim

        self.fusion_dropout = nn.Dropout(0.5)

        if 'vgg' in model_name:
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 1024),    # 4160 → 1024
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024, 256),           # 1024 → 256
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),             # 256 → 64
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)                # 64 → 1
            )
        else:
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, image, metadata):
        # Extract image features
        img_features = self.backbone(image)
        img_features = self.backbone_dropout(img_features)
        
        # Process metadata
        meta_features = self.metadata_net(metadata)
        
        # Fusion
        if self.fusion_strategy == 'attention':
            fused = self.fusion(img_features, meta_features)
        elif self.fusion_strategy == 'concat':
            fused = torch.cat([img_features, meta_features], dim=1)
        else:  # gated fusion
            concat_features = torch.cat([img_features, meta_features], dim=1)
            gate = self.gate(concat_features)
            fused = concat_features * gate

        fused = self.fusion_dropout(fused)
        
        # Classification
        output = self.classifier(fused)
        return output
class StratifiedGroupCV:
    """StratifiedGroupKFold wrapper - prevents patient leakage"""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
    
    def split(self, df):
        """
        Generate train/val splits with group stratification
        
        Args:
            df: DataFrame with target and group columns
            target_col: Target column name
            group_col: Group column name (patient_id)
        """
        print(f"Creating StratifiedGroupKFold with {self.n_splits} folds (random_state={self.random_state})")
        
        skgf = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        for fold_idx, (train_idx, val_idx) in enumerate(
            skgf.split(df, y=df['target'], groups=df['patient_id'])
        ):
            print(f"\nFold {fold_idx + 1}/{self.n_splits}")
            print(f"  Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")
            print(f"  Train target rate: {df.iloc[train_idx]['target'].mean():.4f}")
            print(f"  Val target rate: {df.iloc[val_idx]['target'].mean():.4f}")
            
            yield train_idx, val_idx
def main_training_pipeline(data_path, num_epochs=30, batch_size=64, num_folds=5):
    """Complete training pipeline"""
    
    print("=" * 80)
    print("SLICE-3D ADVANCED TRAINING PIPELINE")
    print("=" * 80)
    
    # Initialize components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data using your extractor
    extractor = SLICE3DDataExtractor(data_path)
    extractor.extract_data()
    
    train_df = extractor.train_csv
    image_paths = extractor.image_paths
    
    # Calculate class weights
    class_counts = train_df['target'].value_counts()
    total_samples = len(train_df)
    class_weights = {
        0: total_samples / (2 * class_counts[0]),
        1: total_samples / (2 * class_counts[1])
    }
    print(f"Class weights: {class_weights}")
    
    # Stratified K-Fold setup
    sgkf = StratifiedGroupCV(n_splits=num_folds, random_state=42)
    
    # Store models for ensemble
    all_models = []
    fold_scores = []
    fold_loss_history = {}
    
    # Model configurations
    model_configs = [
        {'name': 'resnet18', 'fusion': 'concat'},
        {'name': 'vgg16', 'fusion': 'concat'},
        {'name': 'resnet34', 'fusion': 'gated'},
        {'name': 'efficientnet_b0', 'fusion': 'attention'},
        {'name': 'efficientnet_b1', 'fusion': 'concat'},
    ]
    
    # Training loop
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(train_df)):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold + 1}/{num_folds}")
        print(f"{'=' * 60}")
        
        # Split data
        train_data = train_df.iloc[train_idx].copy()
        val_data = train_df.iloc[val_idx].copy()

        print(f"\nEngineering features for fold {fold + 1}...")

        feature_engineer = FeatureEngineer()

        train_meta = feature_engineer.create_features(train_data, is_training=True)
        print(f"Training features shape: {train_meta.shape}")

        val_meta = feature_engineer.create_features(val_data, is_training=False)
        print(f"Validation features shape: {val_meta.shape}")
        
        # Create datasets
        preprocessor = ImagePreprocessor()
        
        train_dataset = SLICE3DDataset(
            train_data, image_paths, train_meta,
            preprocessor=preprocessor,
            is_training=True
        )
        
        val_dataset = SLICE3DDataset(
            val_data, image_paths, val_meta,
            preprocessor=preprocessor,
            is_training=False
        )
        
        # Create dataloaders with weighted sampling
        from torch.utils.data.sampler import WeightedRandomSampler
        
        sample_weights = [class_weights[int(t)] for t in train_data['target'].values]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            sampler=sampler, num_workers=4, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        # Train multiple models per fold
        fold_models = []
        
        for config in model_configs[:1]:  # Use 2 models per fold for efficiency
            print(f"\nTraining {config['name']} with {config['fusion']} fusion...")
            
            # Initialize model
            model = SLICE3DModel(
                model_name=config['name'],
                metadata_dim=train_meta.shape[1],
                dropout=0.5,
                fusion_strategy=config['fusion']
            )
            
            # Initialize trainer
            trainer = Trainer(model, device, learning_rate=1e-4, weight_decay=1e-5)
            
            # Training epochs
            best_val_pauc = 0
            train_losses = []
            val_losses = []
            best_metrics = {'acc': 0, 'prec': 0, 'rec': 0, 'pauc': 0}
            
            for epoch in range(num_epochs):
                
                if epoch == 5:
                    if hasattr(model, 'unfreeze_backbone'):
                        model.unfreeze_backbone()
                        print("Unfreezing backbone layers...")
                        # Reduce learning rate when unfreezing
                        for param_group in trainer.optimizer.param_groups:
                           param_group['lr'] *= 0.5
                
                # Train
                train_loss, train_pauc, train_acc, train_prec, train_rec = trainer.train_epoch(train_loader, class_weights=None)
                # Validate
                val_loss, val_pauc, val_acc, val_prec, val_rec = trainer.validate(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Update scheduler
                trainer.scheduler.step()

                print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train pAUC: {train_pauc:.4f}, Val pAUC: {val_pauc:.4f}, "
                      f"Train Acc: {train_acc:.4f}, Train Prec: {train_prec:.4f}, Train Recall: {train_rec:.4f}"
                      f"Val Acc: {val_acc:.4f}, Val Prec: {val_prec:.4f}, Val Recall: {val_rec:.4f}")
                
                if val_pauc > best_val_pauc:
                    best_val_pauc = val_pauc
                    # Save best model
                    torch.save(model.state_dict(), 
                              f'best_model_fold{fold}_{config["name"]}.pth')
                    best_metrics = {
                        'acc': val_acc,
                        'prec': val_prec,
                        'rec': val_rec,
                        'pauc': val_pauc
                    }
            
            # Load best model
            model.load_state_dict(torch.load(f'best_model_fold{fold}_{config["name"]}.pth'))
            fold_models.append(model)
            print(f"Best validation pAUC: {best_val_pauc:.4f}")
        
        all_models.extend(fold_models)
        fold_scores.append(best_val_pauc)
        print(f"\n========== FOLD {fold+1} BEST METRICS ==========")
        print(f"Best Accuracy : {best_metrics['acc']:.4f}")
        print(f"Best Precision: {best_metrics['prec']:.4f}")
        print(f"Best Recall   : {best_metrics['rec']:.4f}")
        print(f"Best pAUC     : {best_metrics['pauc']:.4f}")
        print("===============================================\n")

    
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"Average validation pAUC: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    print(f"{'=' * 60}")
    
    return all_models, feature_engineer
if __name__ == "__main__":
    input_path = "./dataset/isic-2024-challenge/data/"

    models, feature_engineer = main_training_pipeline(
        data_path = input_path ,
        num_epochs = 100,
        batch_size = 165,
        num_folds = 5
    )
