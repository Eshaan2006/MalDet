import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from joblib import dump
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the PE feature extractor (assuming it's in the same directory)
from extract_features import PEFeatureExtractor

class DikeDatasetProcessor:
    def __init__(self, dataset_path="/home/ubuntu/mal-dec/DikeDataset/files"): # Update this to your dataset path
        self.dataset_path = dataset_path
        self.benign_path = os.path.join(dataset_path, "benign")
        self.malware_path = os.path.join(dataset_path, "malware")
        self.extractor = PEFeatureExtractor()
        self.scaler = StandardScaler()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def scan_dataset(self):
        """Scan dataset to understand the distribution"""
        benign_files = []
        malware_files = []
        
        if os.path.exists(self.benign_path):
            benign_files = [f for f in os.listdir(self.benign_path) 
                           if f.lower().endswith(('.exe', '.dll', '.sys'))]
        
        if os.path.exists(self.malware_path):
            malware_files = [f for f in os.listdir(self.malware_path) 
                           if f.lower().endswith(('.exe', '.dll', '.sys'))]
        
        self.logger.info(f"Benign: {len(benign_files)}, Malware: {len(malware_files)}")
        
        return benign_files, malware_files

    def extract_features_from_dataset(self, max_samples_per_class=None, sample_malware=True):
        """Extract features from all files in the dataset"""
        benign_files, malware_files = self.scan_dataset()
        
        # Sample files if requested (useful for testing)
        if max_samples_per_class:
            benign_files = benign_files[:max_samples_per_class]
            np.random.seed(42)
            if sample_malware:
                # Sample malware to match benign count for initial testing
                
                malware_files = np.random.choice(malware_files, 
                                               min(len(malware_files), max_samples_per_class), 
                                               replace=False).tolist()
        
        features_list = []
        labels = []
        file_paths = []
        
        # Process benign files
        for label, file_list, path in [(0, benign_files, self.benign_path), (1, malware_files, self.malware_path)]:
            desc = "Benign files" if label == 0 else "Malware files"
            self.logger.info(f"Processing {desc}...")
            for fn in tqdm(file_list, desc=desc):
                fp = os.path.join(path, fn)
                feat = self.extractor.extract_features(fp)
                if feat is not None:
                    features_list.append(feat)
                    labels.append(label)
                    file_paths.append(fp)

        df = pd.DataFrame(features_list)
        df['label'] = labels
        df['file_path'] = file_paths
        self.logger.info(f"Total processed: {len(df)} files")
        return df

    def preprocess_features(self, df, handle_missing='median'):
        """Preprocess features: handle missing values, encode categoricals, scale"""
        self.logger.info("Preprocessing features...")
        
        # Separate features from labels and metadata
        feature_cols = [c for c in df.columns if c not in ['label', 'file_path']]
        X = df[feature_cols].copy()
        y = df['label'].astype(int)
        
        # Handle missing values
        if handle_missing == 'median':
            X = X.fillna(X.median())
        elif handle_missing == 'mean':
            X = X.fillna(X.mean())
        else:
            X = X.fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        self.logger.info(f"Feature summary before scaling:\n{X.describe().T}")  
        return X, y

    def apply_smote(self, X, y, method='smote', sampling_strategy='auto', random_state=42):
        """Apply SMOTE or its variants to balance the dataset"""
        self.logger.info(f"Applying {method} oversampling")
        
        # Choose SMOTE variant
        if method == 'smote':
            sampler = SMOTE(sampling_strategy=sampling_strategy, 
                          random_state=random_state, 
                          )
        elif method == 'adasyn':
            sampler = ADASYN(sampling_strategy=sampling_strategy, 
                           random_state=random_state)
        elif method == 'borderline':
            sampler = BorderlineSMOTE(sampling_strategy=sampling_strategy, 
                                    random_state=random_state)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(sampling_strategy=sampling_strategy, 
                               random_state=random_state)
        elif method == 'smote_enn':
            sampler = SMOTEENN(sampling_strategy=sampling_strategy, 
                             random_state=random_state)
        else:
            raise ValueError(f"Unknown SMOTE method: {method}")
        
        self.smote_sampler = sampler
        try:
            X_res, y_res = sampler.fit_resample(X, y)
            self.logger.info(f"New class distribution: {np.bincount(y_res)}")
            self.logger.info(f"Feature summary after SMOTE:\n{pd.DataFrame(X_res, columns=X.columns).describe().T}")
            return X_res, y_res
        except Exception as e:
            self.logger.error(f"SMOTE failed: {e}")
            return X, y

    def create_train_test_split(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Create train/validation/test splits"""
        self.logger.info("Splitting dataset...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
            stratify=y_temp
        )
        
        self.logger.info(f"Train/Val/Test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(self, X_train, X_val, X_test):
        """Scale features using StandardScaler fitted on training data"""
        self.logger.info("Scaling features...")
        
        # Fit scaler on training data only
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        X_test_s = self.scaler.transform(X_test)
        
        return X_train_s, X_val_s, X_test_s

    def process_complete_pipeline(self, max_samples_per_class=None, 
                                smote_method='smote', 
                                sampling_strategy='auto',
                                save_processed_data=True):
        """Complete processing pipeline"""
        self.logger.info("Starting complete processing pipeline...")
        
        # Step 1: Extract features
        df = self.extract_features_from_dataset(max_samples_per_class)
        
        # Step 2: Preprocess features
        X, y = self.preprocess_features(df)
        
        # Step 3: Create initial train/test split (before SMOTE)
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_train_test_split(X, y)
        
        # Step 4: Apply SMOTE only to training data
        X_train_balanced, y_train_balanced = self.apply_smote(
            X_train, y_train, method=smote_method, sampling_strategy=sampling_strategy
        )
        
        # Step 5: Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train_balanced, X_val, X_test
        )
        
        # Prepare final datasets
        datasets = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_balanced,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'scaler': self.scaler
        }
        
        # Save processed data
        if save_processed_data:
            self.save_processed_datasets(datasets, df)
        
        self.logger.info("Pipeline completed successfully!")
        return datasets

    def save_processed_datasets(self, datasets, original_df):
        """Save processed datasets to disk"""
        save_dir = "processed_data"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save datasets
        dump(datasets, os.path.join(save_dir, "processed_datasets.pkl"))
        # Save individual artifacts
        dump(self.scaler, os.path.join(save_dir, "scaler.pkl"))
        dump(self.smote_sampler, os.path.join(save_dir, "smote_sampler.pkl"))
        original_df.to_csv(os.path.join(save_dir, "extracted_features.csv"), index=False)
        with open(os.path.join(save_dir, "feature_names.txt"), 'w') as f:
            for feat in datasets['feature_names']:
                f.write(feat + "\n")
        self.logger.info(f"Artifacts saved to {save_dir}/")

    def load_processed_datasets(self, path="processed_data/processed_datasets.pkl"):
        """Load previously processed datasets"""
        if os.path.exists(path):
            datasets = joblib.load(path)
            self.scaler = datasets['scaler']  # Restore scaler
            self.logger.info("Loaded processed datasets from disk")
            return datasets
        else:
            raise FileNotFoundError(f"No processed datasets found at {path}")

    def get_dataset_statistics(self, datasets):
        """Print comprehensive dataset statistics"""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        print(f"Training set: {datasets['X_train'].shape}")
        print(f"Validation set: {datasets['X_val'].shape}")
        print(f"Test set: {datasets['X_test'].shape}")
        print(f"Number of features: {len(datasets['feature_names'])}")
        
        print(f"\nClass distribution (Training):")
        train_dist = np.bincount(datasets['y_train'])
        print(f"  Benign: {train_dist[0]} ({train_dist[0]/len(datasets['y_train'])*100:.1f}%)")
        print(f"  Malware: {train_dist[1]} ({train_dist[1]/len(datasets['y_train'])*100:.1f}%)")
        
        print(f"\nClass distribution (Test):")
        test_dist = np.bincount(datasets['y_test'])
        print(f"  Benign: {test_dist[0]} ({test_dist[0]/len(datasets['y_test'])*100:.1f}%)")
        print(f"  Malware: {test_dist[1]} ({test_dist[1]/len(datasets['y_test'])*100:.1f}%)")
        
        print(f"\nFeature statistics:")
        X_train = datasets['X_train']
        print(f"  Mean feature value: {np.mean(X_train):.3f}")
        print(f"  Std feature value: {np.std(X_train):.3f}")
        print(f"  Min feature value: {np.min(X_train):.3f}")
        print(f"  Max feature value: {np.max(X_train):.3f}" )

    def process_complete_pipeline(self, max_samples_per_class=None, smote_method='smote', sampling_strategy='auto', save_processed_data=True):
        df = self.extract_features_from_dataset(max_samples_per_class)
        X, y = self.preprocess_features(df)
        X_tr, X_val, X_te, y_tr, y_val, y_te = self.create_train_test_split(X, y)
        X_tr_b, y_tr_b = self.apply_smote(X_tr, y_tr, method=smote_method, sampling_strategy=sampling_strategy)
        X_tr_s, X_val_s, X_te_s = self.scale_features(X_tr_b, X_val, X_te)

        datasets = {
            'X_train': X_tr_s, 'X_val': X_val_s, 'X_test': X_te_s,
            'y_train': y_tr_b, 'y_val': y_val, 'y_test': y_te,
            'feature_names': X.columns.tolist(), 'scaler': self.scaler, 'smote_sampler': self.smote_sampler
        }

        if save_processed_data:
            self.save_processed_datasets(datasets, df)

        self.logger.info("Pipeline completed successfully!")
        return datasets


def main():
    """Main processing function"""
    # Initialize processor
    processor = DikeDatasetProcessor()
    
    # For initial testing, process a smaller subset
    # Remove max_samples_per_class=None for full dataset
    datasets = processor.process_complete_pipeline(
        smote_method='smote',  # Options: 'smote', 'adasyn', 'borderline', 'smote_tomek'
        sampling_strategy='auto',  # 'auto' for full balancing, or specify ratio like 0.8
        save_processed_data=True
    )
    
    # Print statistics
    processor.get_dataset_statistics(datasets)
    
    return datasets

if __name__ == "__main__": 
    main()
