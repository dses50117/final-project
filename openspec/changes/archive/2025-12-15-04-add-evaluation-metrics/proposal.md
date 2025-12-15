# 04 - Add Evaluation Metrics and Threshold Tuning

**Status**: Proposed  
**Created**: 2025-12-15  
**Author**: dses50117

## Summary

Implement comprehensive model evaluation with multiple metrics, threshold tuning, and automated reporting to ensure the model meets business requirements.

## Motivation

Based on CRISP-DM Phase 5 (Evaluation), we need to:
- Measure model performance beyond simple accuracy
- Optimize the classification threshold (not default 0.5)
- Generate visual reports (confusion matrix, classification report)
- Export results for stakeholder review

**Why threshold tuning matters**: For fatigue detection, we may prioritize **high Recall** (catch all drowsy cases) over Precision (some false alarms are acceptable).

## Changes Made

### Key Features
- **Multi-Metric Evaluation**:
  - Accuracy
  - Precision (avoid false alarms)
  - Recall (catch all drowsy cases)
  - F1-Score (harmonic mean)
  - ROC-AUC (model discrimination ability)
  
- **Threshold Tuning**: 
  - Search range: 0.4 to 0.8
  - Find threshold that maximizes accuracy
  - Balance between false positives and false negatives
  
- **Visualization**:
  - Confusion Matrix heatmap
  - Classification Report table
  
- **Export Reports**:
  - `metrics.xlsx` - Excel report with all metrics
  - `metrics.csv` - CSV version for scripting
  - `confusion_matrix.png` - Visual confusion matrix

### Files to Modify
- `train_model.py` - Add evaluation functions

### Core Functions
```python
def evaluate_model(model, X_test, y_test):
    """Calculate all metrics"""
    
def tune_threshold(model, X_test, y_test):
    """Find optimal classification threshold"""
    
def plot_confusion_matrix(y_true, y_pred):
    """Generate confusion matrix visualization"""
    
def export_metrics(metrics_dict):
    """Save metrics to Excel/CSV"""
```

### Evaluation Output Format
```
Model: XGBoost
Accuracy: 0.92
Precision: 0.89
Recall: 0.95
F1-Score: 0.92
ROC-AUC: 0.96
Optimal Threshold: 0.45

Confusion Matrix:
                Predicted
              NotDrowsy  Drowsy
Actual
NotDrowsy        850      50
Drowsy           30       570
```

## Impact
- **Affected specs**: None
- **Affected code**: Extends `train_model.py`
- **Dependencies**: `scikit-learn`, `matplotlib`, `seaborn`, `openpyxl`

## Tasks

### 1. Metrics Implementation
- [ ] Implement multi-metric calculation
- [ ] Add ROC-AUC scoring
- [ ] Create classification report generator

### 2. Threshold Tuning
- [ ] Implement threshold search (0.4-0.8)
- [ ] Test with validation data
- [ ] Save optimal threshold to config

### 3. Visualization
- [ ] Create confusion matrix plot
- [ ] Add metric comparison charts
- [ ] Export plots to PNG

### 4. Reporting
- [ ] Export to Excel format
- [ ] Export to CSV format
- [ ] Create automated report template
