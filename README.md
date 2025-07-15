
The pipeline takes raw data, processes it, extracts features, trains models, and evaluates them. It supports both traditional machine learning models (such as XGBoost) and deep learning models (such as CNNs and GNNs). It also includes utilities for model pruning to reduce size and improve efficiency, and for threshold tuning to optimize classification performance.

## Workflow

1. Data Preprocessing

Script: data_processor.py  
Purpose: Cleans raw data, handles missing values, normalizes or scales features, and splits data into train/test sets.  
Output: Preprocessed data files ready for feature extraction.

2. Feature Extraction

Script: extract_features.py  
Purpose: Generates feature matrices from preprocessed data. This can include manual feature engineering or automatic feature selection.  
Output: Feature sets used as input for model training.

3. Model Training

- XGBoost (traditional ML):
  Script: train_xgboost.py
  Trains an XGBoost model using the extracted features.

- Deep learning (PyTorch):
  Script: tier2_pytorch.py
  Defines and trains CNN and GNN model.

4. Model Pruning and Retraining (optional)

Script: prune_and_retrain.py  
Purpose: Applies pruning techniques to deep learning models to reduce model complexity and size. Retrains the pruned model to recover performance.

5. Threshold Selection

Script: select_threshold.py  
Purpose: Selects an optimal threshold for converting predicted probabilities into final class labels, for example to improve F1-score or balance precision and recall.

6. Model Testing

- CNN:
  Script: test_cnn.py
  Evaluates a trained CNN on test data.

- GNN:
  Script: test_gnn.py
  Evaluates a GNN on graph-based data.

- Testing for XGBoost:
  Script: test_sample.py
  Evaluates based on XGboost model.

## Requirements

- Python >= 3.7
- PyTorch
- XGBoost
- pandas
- NumPy
- scikit-learn
- (Optional) PyTorch Geometric for GNN support

## Usage

Example order to run:

1. python data_processor.py
2. python extract_features.py
3. python train_xgboost.py
4. python prune_and_retrain.py      # optional
5. python select_threshold.py
6. python test_cnn.py
7. python test_gnn.py
8. python test_sample.py

