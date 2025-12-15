# 05 - Deploy Adaptive Calibration System

**Status**: Proposed  
**Created**: 2025-12-15  
**Author**: dses50117

## Summary

Implement adaptive calibration that personalizes EAR thresholds for each user, plus add dual verification (ML + Rule-based) for robust fatigue detection in production.

## Motivation

Based on CRISP-DM Phase 6 (Deployment - Advanced Features), we need to solve a critical real-world problem:

**Problem**: Different people have different eye shapes and sizes. A fixed EAR threshold (e.g., 0.25) may:
- Be too sensitive for people with smaller eyes (false alarms)
- Be too lenient for people with larger eyes (missed detections)

**Solution**: Adaptive calibration collects each user's baseline EAR during startup and dynamically sets their personal threshold.

## Changes Made

### Key Features

#### 1. Adaptive Calibration Mode
- **Startup Phase**: Collect 30-60 frames of user's normal (awake) EAR values
- **Baseline Calculation**: Calculate mean and std of baseline EAR
- **Dynamic Threshold**: Set threshold = baseline_mean - (2 × std)
- **User Feedback**: Display calibration progress with visual indicators

#### 2. Dual Verification System
**Why**: Combining ML predictions with rule-based checks improves reliability.

- **ML Prediction**: Model predicts drowsy/notdrowsy from features
- **Rule-Based Check**: 
  - If EAR < threshold for N consecutive frames → Drowsy
  - If MAR > yawn_threshold → Yawning (indicator)
- **Final Decision**: Alert if BOTH agree (reduces false positives)

#### 3. Real-time Processing with WebRTC
- **Technology**: `streamlit-webrtc` for async frame processing
- **Processor Class**: `DrowsinessProcessor` handles frame-by-frame logic
- **Threading**: Non-blocking video processing

### Files to Modify
- `app.py` - Add calibration mode and dual verification logic
- `config.py` - Add calibration parameters

### Implementation Details

```python
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.calibration_mode = True
        self.baseline_ears = []
        self.personal_threshold = None
        
    def calibrate(self, ear_value):
        """Collect baseline EAR values"""
        if len(self.baseline_ears) < 60:
            self.baseline_ears.append(ear_value)
        else:
            mean_ear = np.mean(self.baseline_ears)
            std_ear = np.std(self.baseline_ears)
            self.personal_threshold = mean_ear - (2 * std_ear)
            self.calibration_mode = False
            
    def detect_fatigue(self, features):
        """Dual verification: ML + Rules"""
        # ML prediction
        ml_prediction = self.model.predict(features)
        
        # Rule-based check
        rule_check = (features['ear'] < self.personal_threshold)
        
        # Final decision
        return ml_prediction == 'drowsy' and rule_check
```

### Calibration UI Flow
```
1. App Start → "Calibration Mode: Please look at the camera normally"
2. Progress Bar: [=====>      ] 20/60 frames
3. Complete → "✓ Calibration Complete! Your threshold: 0.23"
4. Switch to Detection Mode
```

## Impact
- **Affected specs**: None
- **Affected code**: Major enhancement to `app.py`
- **Dependencies**: `streamlit-webrtc`, `av`
- **User Experience**: Significantly reduces false alarms

## Tasks

### 1. Calibration System
- [ ] Implement baseline EAR collection
- [ ] Add calibration UI with progress bar
- [ ] Calculate personal threshold
- [ ] Test with multiple users

### 2. Dual Verification
- [ ] Integrate ML prediction
- [ ] Add rule-based checks
- [ ] Combine both for final decision
- [ ] Tune consecutive frame threshold

### 3. WebRTC Integration
- [ ] Install streamlit-webrtc
- [ ] Create DrowsinessProcessor class
- [ ] Handle async frame processing
- [ ] Add error handling

### 4. Testing
- [ ] Test calibration with different users
- [ ] Verify threshold accuracy
- [ ] Test dual verification logic
- [ ] Validate real-time performance
