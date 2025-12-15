# 02 - Create Feature Extraction Module

**Status**: Proposed  
**Created**: 2025-12-15  
**Author**: dses50117

## Summary

Implement the feature extraction module that calculates EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), and Head Pose from facial landmarks for fatigue detection.

## Motivation

Based on CRISP-DM Phase 3 (Data Preparation - Feature Engineering), we need to extract meaningful physical features from raw images. MediaPipe Face Mesh provides 468 facial landmarks, which must be transformed into actionable metrics.

## Changes Made

### Key Features
- **EAR Calculation**: Detect eye closure and blinking
- **MAR Calculation**: Detect yawning behavior  
- **Head Pose Estimation**: Detect head nodding and gaze deviation (Pitch, Yaw, Roll)
- **Batch Processing**: Process multiple images from datasets
- **CSV Export**: Generate training data for modeling

### Files to Create
- `feature_extraction.py` - Main feature extraction module
- `fatigue_detector.py` - Real-time detection wrapper

### Core Functions
```python
def calculate_ear(eye_landmarks): 
    """Calculate Eye Aspect Ratio"""
    
def calculate_mar(mouth_landmarks):
    """Calculate Mouth Aspect Ratio"""
    
def estimate_head_pose(face_landmarks):
    """Calculate Pitch, Yaw, Roll using cv2.solvePnP"""
```

### Feature Specifications
- **EAR**: Ratio of eye vertical distances to horizontal distance
  - Normal: ~0.25-0.35
  - Closed: <0.2
  
- **MAR**: Ratio of mouth vertical distances to horizontal distance
  - Normal: <0.5
  - Yawning: >0.6
  
- **Head Pose**: Angles in degrees
  - Pitch: -90° to +90° (nodding)
  - Yaw: -90° to +90° (left/right turn)
  - Roll: -45° to +45° (tilt)

## Impact
- **Affected specs**: None (new capability)
- **Affected code**: Will be imported by `app.py` and `train_model.py`
- **Data**: Processes `data/drowsy` and `data/notdrowsy` folders

## Tasks

### 1. Implementation
- [ ] Set up MediaPipe Face Mesh
- [ ] Implement EAR calculation
- [ ] Implement MAR calculation
- [ ] Implement Head Pose with solvePnP
- [ ] Add batch processing for datasets
- [ ] Create CSV export functionality

### 2. Testing
- [ ] Test with sample images
- [ ] Validate feature ranges
- [ ] Verify CSV output format
