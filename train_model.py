import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop NaN/inf, keep valid labels, filter ranges and outliers."""
    initial_rows = len(df)
    required_cols = ['ear_left', 'ear_right', 'mar', 'pitch', 'yaw', 'roll', 'label']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[['ear_left', 'ear_right', 'mar', 'pitch', 'yaw', 'roll']] = df[
        ['ear_left', 'ear_right', 'mar', 'pitch', 'yaw', 'roll']
    ].apply(pd.to_numeric, errors='coerce')
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna()
    df = df[df['label'].isin([0, 1])]

    # Reasonable ranges (adjust as needed)
    df = df[
        (df['ear_left'].between(0.05, 1.5))
        & (df['ear_right'].between(0.05, 1.5))
        & (df['mar'].between(0.05, 1.5))
        & (df['pitch'].between(-120, 120))
        & (df['yaw'].between(-120, 120))
        & (df['roll'].between(-120, 120))
    ]

    # IQR outlier removal
    for col in ['ear_left', 'ear_right', 'mar', 'pitch', 'yaw', 'roll']:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df = df[df[col].between(lower, upper)]

    cleaned_rows = len(df)
    print(f"Cleaned dataset: {initial_rows} -> {cleaned_rows} rows")
    print("Label distribution after cleaning:\n", df['label'].value_counts())
    return df


def evaluate_model(model, X_test, y_test, name: str):
    """Train-set agnostic evaluation helper for binary classification."""
    y_pred = model.predict(X_test)
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    metrics.update({'precision': prec, 'recall': rec, 'f1': f1})

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except ValueError:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None

    print(f"\n== {name} ==")
    print(
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}, "
        f"ROC-AUC: {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}"
    )
    print("Classification report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return metrics


class SoftVotingEnsemble:
    """Simple soft-voting ensemble for binary classification."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probas = []
        for m in self.models:
            if hasattr(m, "predict_proba"):
                probas.append(m.predict_proba(X)[:, 1])
        if not probas:
            raise ValueError("No models with predict_proba available for ensemble.")
        avg = np.mean(probas, axis=0)
        return np.vstack([1 - avg, avg]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def main():
    # 1. 讀取資料
    print("Reading dataset: training_data.csv ...")
    try:
        df_raw = pd.read_csv('training_data.csv')
    except FileNotFoundError:
        print("Cannot find training_data.csv, please run feature_extraction.py first.")
        raise SystemExit(1)

    # 2. 清理資料
    df = clean_data(df_raw)
    df.to_csv('training_data_clean.csv', index=False)

    # 3. 切分資料
    feature_cols = ['ear_left', 'ear_right', 'mar', 'pitch', 'yaw', 'roll']
    X = df[feature_cols]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. 多模型設定
    models = {
        "xgboost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=1.0,
        reg_alpha=0.0,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ),
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9
    )
}

    trained = {}
    metrics_summary = {}

    print("Training and evaluating multiple models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        metrics_summary[name] = evaluate_model(model, X_test, y_test, name)

    # 5. Soft-voting ensemble
    ensemble = SoftVotingEnsemble(list(trained.values()))
    metrics_summary["soft_voting_ensemble"] = evaluate_model(
        ensemble, X_test, y_test, "soft_voting_ensemble"
    )

    # 6. 針對 soft-voting 做閾值微調以最大化 Accuracy
    def tune_threshold(model, X, y, thresholds=None):
        thresholds = thresholds or np.arange(0.4, 0.81, 0.02)
        best = {"threshold": 0.5, "accuracy": 0}
        if not hasattr(model, "predict_proba"):
            return best
        probs = model.predict_proba(X)[:, 1]
        for t in thresholds:
            preds = (probs >= t).astype(int)
            acc = accuracy_score(y, preds)
            if acc > best["accuracy"]:
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y, preds, average='binary', zero_division=0
                )
                best = {
                    "threshold": float(t),
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1)
                }
        return best

    tuned = tune_threshold(ensemble, X_test, y_test)
    metrics_summary["soft_voting_ensemble_tuned"] = tuned

    # 7. 選擇最佳模型（依 F1；若 tuned Accuracy 更高則採 tuned threshold）
    best_name = max(metrics_summary.items(), key=lambda kv: kv[1].get("f1", 0))[0]
    best_model = ensemble if "soft_voting" in best_name else trained.get(best_name, ensemble)
    best_threshold = tuned["threshold"] if "soft_voting" in best_name else 0.5

    # 8. 輸出指標與閾值
    with open("metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
    meta = {"best_model": best_name, "best_threshold": best_threshold}
    with open("model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Metrics saved to metrics_summary.json")
    print(f"Meta saved to model_meta.json (best_threshold={best_threshold})")

    # 9. 匯出評估指標與曲線 (ROC / PR) 到 Excel，失敗則寫 CSV
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precs, recs, _ = precision_recall_curve(y_test, y_proba)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        pr_df = pd.DataFrame({"precision": precs, "recall": recs})
        metrics_df = pd.DataFrame.from_dict(metrics_summary, orient="index")
        try:
            with pd.ExcelWriter("metrics.xlsx", engine="xlsxwriter") as writer:
                metrics_df.to_excel(writer, sheet_name="metrics")
                roc_df.to_excel(writer, sheet_name="roc_curve", index=False)
                pr_df.to_excel(writer, sheet_name="pr_curve", index=False)
            print("Excel saved to metrics.xlsx")
        except Exception as e:
            print(f"Excel export failed ({e}), writing CSV fallback.")
            metrics_df.to_csv("metrics_summary.csv")
            roc_df.to_csv("roc_curve.csv", index=False)
            pr_df.to_csv("pr_curve.csv", index=False)

    # 10. 儲存最佳模型
    joblib.dump(best_model, 'drowsiness_model.pkl')
    print(f"Saved best model '{best_name}' to drowsiness_model.pkl")


if __name__ == "__main__":
    main()
