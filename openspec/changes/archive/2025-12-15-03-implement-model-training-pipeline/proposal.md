# 03 - Implement Model Training Pipeline

**Status**: Proposed  
**Created**: 2025-12-15  
**Author**: dses50117

## Summary

Build the machine learning training pipeline with data cleaning, multiple model training, and ensemble learning for drowsiness classification.

## Motivation

Based on CRISP-DM Phase 3 (Data Cleaning) and Phase 4 (Modeling), we need a robust training pipeline that:
- Cleans extracted features (remove NaN, Inf, outliers)
- Trains multiple heterogeneous models
- Uses ensemble learning for better accuracy
- Saves the best-performing model

## Changes Made

### Key Features
- **Data Cleaning**: 
  - Remove NaN and Inf values
  - Filter unreasonable ranges (e.g., Pitch > ±120°)
  - IQR-based outlier removal
  
- **Multiple Models**:
  - XGBoost
  - Random Forest
  - Logistic Regression
  - Gradient Boosting
  
- **Ensemble Learning**: Soft Voting Ensemble for weighted prediction averaging
- **Train/Test Split**: 80/20 with stratified sampling
- **Model Persistence**: Save trained models as .pkl files

### Files to Create
- `train_model.py` - Main training script
- `models/` - Directory for saved models
- `model_meta.json` - Model metadata and parameters

### Core Components
```python
def clean_data(df):
    """Remove NaN, Inf, outliers using IQR method"""
    
class SoftVotingEnsemble:
    """Weighted averaging of model predictions"""
    
def train_models(X_train, y_train):
    """Train all models with hyperparameters"""
    
def save_best_model(models, metrics):
    """Select and save best model by F1-Score"""
```

### Model Hyperparameters
- **XGBoost**: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`
- **RandomForest**: `n_estimators=100`, `max_depth=10`
- **LogisticRegression**: `max_iter=1000`, `C=1.0`
- **GradientBoosting**: `n_estimators=100`, `learning_rate=0.1`

## Impact
- **Affected specs**: None (new capability)
- **Affected code**: 
  - Reads CSV from `feature_extraction.py`
  - Outputs models used by `app.py`
- **Dependencies**: `scikit-learn`, `xgboost`

## Tasks

### 1. Data Cleaning
- [ ] Implement NaN/Inf removal
- [ ] Add range validation
- [ ] Implement IQR outlier detection
- [ ] Test cleaning with sample data

### 2. Model Training
- [ ] Implement XGBoost training
- [ ] Implement Random Forest training
- [ ] Implement Logistic Regression
- [ ] Implement Gradient Boosting
- [ ] Create SoftVotingEnsemble class
- [ ] Add train/test split logic

### 3. Model Persistence
- [ ] Save individual models
- [ ] Save ensemble model
- [ ] Export model metadata JSON
- [ ] Verify model loading
