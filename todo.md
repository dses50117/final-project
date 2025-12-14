# Todo List

## Initial Setup
- [x] Create todo.md file and document initial tasks.

## Project Stages
- [ ] Stage 1: Feature Extraction and Dataset Creation.
- [ ] Stage 2: Model Training.
- [ ] Stage 3: Streamlit Deployment.

## Notes
- Environment: activate with `.\.venv\Scripts\activate`.
- Stage 1: run `python feature_extraction.py` to build `training_data.csv` from `data/`.
- Stage 2: train model with `python train_model.py --data training_data.csv --out drowsiness_model.pkl`.
- Stage 3: launch app with `streamlit run app.py` (uses `st.camera_input` to capture frames).
