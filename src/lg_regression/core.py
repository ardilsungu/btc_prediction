from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path(__file__).parents[2] / "data"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = DATA_DIR / "LRmodels"

CLASS_NAMES = ["DOWN (-1)", "NEUTRAL (0)", "UP (+1)"]
LABEL_MAP = {-1: 0, 0: 1, 1: 2}


def load_split_data(timeframe: str) -> tuple:
    train = pd.read_csv(SPLITS_DIR / f"{timeframe}_train.csv")
    val = pd.read_csv(SPLITS_DIR / f"{timeframe}_val.csv")
    test = pd.read_csv(SPLITS_DIR / f"{timeframe}_test.csv")

    target_col = "Target"
    feature_cols = [c for c in train.columns if c not in (target_col, 'Open time')]


    X_train, y_train = train[feature_cols].values, train[target_col].values
    X_val, y_val = val[feature_cols].values, val[target_col].values
    X_test, y_test = test[feature_cols].values, test[target_col].values

    print(f"[{timeframe}] train={len(X_train)}, val={len(X_val)}, test={len(X_test)}, features={len(feature_cols)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_map = {cls: total / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}
    return np.array([weight_map[label] for label in y])


def train_baseline(X_train, y_train, X_val, y_val, sample_weights):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        random_state=42,
    )
    model.fit(X_train_s, y_train, sample_weight=sample_weights)

    preds = model.predict(X_val_s)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    print(f"[Baseline] Val Accuracy: {acc:.4f} | Val Macro F1: {f1:.4f}")
    return model, scaler


def run_optuna_tuning(X_train, y_train, X_val, y_val, sample_weights, n_trials: int = 50):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        max_iter = trial.suggest_int("max_iter", 500, 3000)
        penalty = trial.suggest_categorical("penalty", ["l2"])

        model = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            C=C,
            penalty=penalty,
            class_weight='balanced',
            random_state=42,
        )
        model.fit(X_train_s, y_train, sample_weight=sample_weights)
        preds = model.predict(X_val_s)
        return f1_score(y_val, preds, average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"[Optuna] Best val Macro F1: {study.best_value:.4f}")
    print(f"[Optuna] Best params: {best_params}")
    return best_params


def train_final_model(X_train, y_train, X_val, y_val, sample_weights, best_params):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        **best_params,
    )
    model.fit(X_train_s, y_train, sample_weight=sample_weights)

    preds = model.predict(X_val_s)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro", zero_division=0)
    print(f"[Final] Val Accuracy: {acc:.4f} | Val Macro F1: {f1:.4f}")
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test, feature_cols, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    X_test_s = scaler.transform(X_test)
    preds = model.predict(X_test_s)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    print(f"[Evaluate] Test Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

    report = classification_report(y_test, preds, target_names=CLASS_NAMES, zero_division=0)
    print(report)

    (output_dir / "classification_report.txt").write_text(report)

    metrics = {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Coefficient importance — top 20 per class
    classes = model.classes_
    coef = model.coef_  # shape (n_classes, n_features)
    n_classes = len(classes)
    fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 8), sharey=False)
    if n_classes == 1:
        axes = [axes]
    for i, (cls, ax) in enumerate(zip(classes, axes)):
        importances = coef[i]
        top_idx = np.argsort(np.abs(importances))[-20:][::-1]
        top_feat = [feature_cols[j] for j in top_idx]
        top_vals = importances[top_idx]
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in top_vals]
        ax.barh(range(len(top_feat)), top_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top_feat)))
        ax.set_yticklabels(top_feat[::-1], fontsize=8)
        ax.set_title(f"Class {cls} Coefficients")
        ax.set_xlabel("Coefficient value")
    plt.suptitle("Top 20 Feature Coefficients per Class", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / "coefficient_importance.png", dpi=150)
    plt.close(fig)

    return metrics


def save_model(model, scaler, best_params: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    (output_dir / "params.json").write_text(json.dumps(best_params, indent=2))
    print(f"[Save] Model saved to {output_dir}")


def load_model(model_dir: Path):
    model = joblib.load(model_dir / "model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    return model, scaler


def full_training_pipeline(
    timeframe: str,
    n_trials: int = 50,
    timeout: int = 1800,
    skip_tuning: bool = False,
):
    print(f"\n{'='*60}")
    print(f"  LR Training Pipeline — {timeframe}")
    print(f"{'='*60}")

    output_dir = MODELS_DIR / timeframe
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_split_data(timeframe)

    # 2. Sample weights
    sample_weights = compute_sample_weights(y_train)

    # 3. Baseline
    print("\n--- Baseline ---")
    train_baseline(X_train, y_train, X_val, y_val, sample_weights)

    # 4. Optuna tuning
    if skip_tuning:
        print("\n--- Skipping Optuna (--skip-tuning) ---")
        best_params = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "penalty": "l2"}
    else:
        print(f"\n--- Optuna ({n_trials} trials) ---")
        best_params = run_optuna_tuning(X_train, y_train, X_val, y_val, sample_weights, n_trials=n_trials)

    # 5. Final model
    print("\n--- Final Model ---")
    model, scaler = train_final_model(X_train, y_train, X_val, y_val, sample_weights, best_params)

    # 6. Evaluate
    print("\n--- Evaluation ---")
    evaluate_model(model, scaler, X_test, y_test, feature_cols, output_dir)

    # 7. Save
    save_model(model, scaler, best_params, output_dir)

    print(f"\n[Done] Artifacts saved to {output_dir}")
    return model, scaler, best_params
