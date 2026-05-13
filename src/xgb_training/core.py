import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI olmadan plot uretmek icin
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

# Optuna log seviyesini dusur (cok fazla cikti uretmesin)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPLIT_DIR    = PROJECT_ROOT / 'data' / 'splits'
MODEL_DIR    = PROJECT_ROOT / 'data' / 'XGmodels'


# --------------------------------------------------------------
# VERI YUKLEME
# --------------------------------------------------------------
def load_split_data(timeframe: str) -> tuple:
    """
    Split edilmis CSV dosyalarini yukler.

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
    """
    train_path = SPLIT_DIR / f'{timeframe}_train.csv'
    val_path   = SPLIT_DIR / f'{timeframe}_val.csv'
    test_path  = SPLIT_DIR / f'{timeframe}_test.csv'

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Split dosyasi bulunamadi: {p}\n"
                f"Once time_split.py calistirin: python time_split.py --timeframe {timeframe}"
            )

    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    # Feature kolonlari = Target ve Open time haric her sey
    feature_cols = [c for c in train_df.columns if c not in ('Open time', 'Target')]

    X_train = train_df[feature_cols].values
    y_train = train_df['Target'].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df['Target'].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df['Target'].values

    print(f"  Train      : {X_train.shape[0]:>8,} satir x {X_train.shape[1]} feature")
    print(f"  Validation : {X_val.shape[0]:>8,} satir x {X_val.shape[1]} feature")
    print(f"  Test       : {X_test.shape[0]:>8,} satir x {X_test.shape[1]} feature")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


# --------------------------------------------------------------
# SINIF AGIRLIK HESABI
# --------------------------------------------------------------
def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Sinif dengesizligini telafi eden per-sample weight hesaplar.
    Formul: weight_i = total / (n_classes * count_class_i)
    """
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    total = len(y)

    class_weights = {}
    for cls, cnt in zip(classes, counts):
        class_weights[cls] = total / (n_classes * cnt)

    if -1 in class_weights: class_weights[-1] *= 1.8
    if 1 in class_weights: class_weights[1] *= 1.8
    if 0 in class_weights: class_weights[0] *= 0.6

    weights = np.array([class_weights[yi] for yi in y])

    print(f"\n  Sinif agirliklari:")
    for cls in sorted(class_weights):
        label = {-1: 'DOWN', 0: 'NEUTRAL', 1: 'UP'}.get(cls, str(cls))
        print(f"    {label:>8s} (class={cls:>2d}) : {class_weights[cls]:.4f}")

    return weights


# --------------------------------------------------------------
# TARGET LABEL MAPPING  (-1, 0, 1) -> (0, 1, 2)
# --------------------------------------------------------------
def encode_labels(y: np.ndarray) -> np.ndarray:
    """XGBoost multi-class 0'dan baslayan label ister: -1->0, 0->1, 1->2"""
    return y + 1


def decode_labels(y_encoded: np.ndarray) -> np.ndarray:
    """Encoded labellari orijinal degerlere cevirir: 0->-1, 1->0, 2->1"""
    return y_encoded - 1


# --------------------------------------------------------------
# BASELINE MODEL
# --------------------------------------------------------------
def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: np.ndarray,
) -> xgb.XGBClassifier:
    """
    Varsayilan parametrelerle baseline XGBoost modeli egitir.
    Early stopping ile overfitting onlenir.
    """
    print(f"\n{'-'*60}")
    print("  BASELINE MODEL EGITIMI")
    print(f"{'-'*60}")

    y_train_enc = encode_labels(y_train)
    y_val_enc   = encode_labels(y_val)

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        device='cuda',
        tree_method='hist',
        eval_metric='mlogloss',
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        sample_weight=sample_weights,
        verbose=False,
    )

    # Baseline sonuclari
    y_pred_enc = model.predict(X_val)
    y_pred = decode_labels(y_pred_enc)

    f1_macro = f1_score(y_val, y_pred, average='macro')
    acc      = accuracy_score(y_val, y_pred)

    print(f"\n  Baseline Sonuclari (Validation Set):")
    print(f"    Accuracy   : {acc:.4f}")
    print(f"    Macro F1   : {f1_macro:.4f}")
    print(f"    Best iter  : {model.best_iteration}")

    return model


# --------------------------------------------------------------
# OPTUNA HYPERPARAMETER TUNING
# --------------------------------------------------------------
def run_optuna_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: np.ndarray,
    n_trials: int = 50,
    timeout: int = 1800,  # 30 dakika
) -> dict:
    """
    Optuna ile Bayesian hyperparameter optimizasyonu.

    Parameters
    ----------
    n_trials : Denenecek parametre kombinasyonu sayisi.
    timeout  : Maksimum sure (saniye). Sure dolunca durur.

    Returns
    -------
    best_params : En iyi hyperparametreler.
    """
    print(f"\n{'-'*60}")
    print(f"  OPTUNA TUNING ({n_trials} trial, timeout={timeout}s)")
    print(f"{'-'*60}")

    y_train_enc = encode_labels(y_train)
    y_val_enc   = encode_labels(y_val)

    def objective(trial):
        params = {
            'objective':        'multi:softmax',
            'num_class':        3,
            'device':           'cuda',
            'tree_method':      'hist',
            'eval_metric':      'mlogloss',
            'random_state':     42,
            'verbosity':        0,
            'early_stopping_rounds': 30,

            # Tune edilecek parametreler
            'n_estimators':     trial.suggest_int('n_estimators', 200, 1500),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma':            trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.5, 5.0),
        }

        model = xgb.XGBClassifier(**params)

        model.fit(
            X_train, y_train_enc,
            eval_set=[(X_val, y_val_enc)],
            sample_weight=sample_weights,
            verbose=False,
        )

        y_pred_enc = model.predict(X_val)
        y_pred = decode_labels(y_pred_enc)
        return f1_score(y_val, y_pred, average='macro')

    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_btc',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n  En iyi trial: #{best.number}")
    print(f"  En iyi Macro F1: {best.value:.4f}")
    print(f"  Parametreler:")
    for k, v in best.params.items():
        print(f"    {k:>22s} : {v}")

    return best.params


# --------------------------------------------------------------
# BINARY RELEVANCE MODELS
# --------------------------------------------------------------
def optimize_threshold_balanced(y_true, y_proba, target_recall=0.40, min_precision=0.30):
    """
    Hem recall hem precision'ı dengeleyen threshold.
    """
    from sklearn.metrics import precision_score, recall_score
    
    thresholds = np.linspace(0.2, 0.9, 100)
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        
        if precision >= min_precision and recall >= target_recall * 0.8:
            score = 2 * (precision * recall) / (precision + recall + 1e-8)
            if score > best_score:
                best_score = score
                best_threshold = thresh
    
    return best_threshold

def train_binary_relevance_models_v2(
    X_train, y_train, X_val, y_val, X_test, y_test,
    sample_weights, best_params
):
    """
    Geliştirilmiş binary relevance - confidence filtering ile.
    """
    print(f"\n{'='*70}")
    print("  BINARY RELEVANCE V2 (BALANCED)")
    print(f"{'='*70}")
    
    print("\n[1/2] UP Detection Model")
    
    y_train_up = (y_train == 1).astype(int)
    y_val_up = (y_val == 1).astype(int)
    y_test_up = (y_test == 1).astype(int)
    
    weights_up = sample_weights.copy()
    weights_up[y_train == 1] *= 2.5
    
    model_up = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=np.sum(y_train_up == 0) / np.sum(y_train_up == 1) if np.sum(y_train_up == 1) > 0 else 1,
        device='cuda',
        tree_method='hist',
        eval_metric='auc',
        n_estimators=best_params.get('n_estimators', 500),
        max_depth=best_params.get('max_depth', 6),
        learning_rate=best_params.get('learning_rate', 0.1),
        subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        random_state=42,
        verbosity=0,
        early_stopping_rounds=30,
    )
    
    model_up.fit(
        X_train, y_train_up,
        eval_set=[(X_val, y_val_up)],
        sample_weight=weights_up,
        verbose=False
    )
    
    val_proba_up = model_up.predict_proba(X_val)[:, 1]
    thresh_up = optimize_threshold_balanced(
        y_val_up, val_proba_up, 
        target_recall=0.35,
        min_precision=0.15
    )
    print(f"  UP threshold (balanced): {thresh_up:.3f}")
    
    val_pred_up = (val_proba_up >= thresh_up).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    val_precision_up = precision_score(y_val_up, val_pred_up, zero_division=0)
    val_recall_up = recall_score(y_val_up, val_pred_up, zero_division=0)
    val_f1_up = f1_score(y_val_up, val_pred_up, zero_division=0)
    print(f"  UP Validation - Precision: {val_precision_up:.3f}, Recall: {val_recall_up:.3f}, F1: {val_f1_up:.3f}")
    
    print("\n[2/2] DOWN Detection Model")
    
    y_train_down = (y_train == -1).astype(int)
    y_val_down = (y_val == -1).astype(int)
    y_test_down = (y_test == -1).astype(int)
    
    weights_down = sample_weights.copy()
    weights_down[y_train == -1] *= 2.5
    
    model_down = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=np.sum(y_train_down == 0) / np.sum(y_train_down == 1) if np.sum(y_train_down == 1) > 0 else 1,
        device='cuda',
        tree_method='hist',
        eval_metric='auc',
        n_estimators=best_params.get('n_estimators', 500),
        max_depth=best_params.get('max_depth', 6),
        learning_rate=best_params.get('learning_rate', 0.1),
        subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        random_state=42,
        verbosity=0,
        early_stopping_rounds=30,
    )
    
    model_down.fit(
        X_train, y_train_down,
        eval_set=[(X_val, y_val_down)],
        sample_weight=weights_down,
        verbose=False
    )
    
    val_proba_down = model_down.predict_proba(X_val)[:, 1]
    thresh_down = optimize_threshold_balanced(
        y_val_down, val_proba_down, 
        target_recall=0.35,
        min_precision=0.15
    )
    print(f"  DOWN threshold (balanced): {thresh_down:.3f}")
    
    val_pred_down = (val_proba_down >= thresh_down).astype(int)
    val_precision_down = precision_score(y_val_down, val_pred_down, zero_division=0)
    val_recall_down = recall_score(y_val_down, val_pred_down, zero_division=0)
    val_f1_down = f1_score(y_val_down, val_pred_down, zero_division=0)
    print(f"  DOWN Validation - Precision: {val_precision_down:.3f}, Recall: {val_recall_down:.3f}, F1: {val_f1_down:.3f}")
    
    print("\n[3/3] Combining Predictions (with confidence filter)")
    
    test_proba_up = model_up.predict_proba(X_test)[:, 1]
    test_proba_down = model_down.predict_proba(X_test)[:, 1]
    
    final_pred = np.zeros(len(y_test), dtype=int)
    
    MIN_CONFIDENCE_GAP = 0.05
    
    up_mask = test_proba_up >= thresh_up
    down_mask = test_proba_down >= thresh_down
    
    final_pred[up_mask & ~down_mask] = 1
    final_pred[down_mask & ~up_mask] = -1
    
    conflict_mask = up_mask & down_mask
    if conflict_mask.sum() > 0:
        up_conf = test_proba_up[conflict_mask]
        down_conf = test_proba_down[conflict_mask]
        
        confident_up = (up_conf > down_conf + MIN_CONFIDENCE_GAP)
        confident_down = (down_conf > up_conf + MIN_CONFIDENCE_GAP)
        
        conflict_indices = np.where(conflict_mask)[0]
        final_pred[conflict_indices[confident_up]] = 1
        final_pred[conflict_indices[confident_down]] = -1
    
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    print("\n" + "="*70)
    print("  BINARY RELEVANCE V2 - TEST RESULTS")
    print("="*70)
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, final_pred,
        target_names=['DOWN (-1)', 'NEUTRAL (0)', 'UP (+1)'],
        digits=4
    ))
    
    acc = accuracy_score(y_test, final_pred)
    f1_macro = f1_score(y_test, final_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, final_pred, average='weighted', zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    
    for cls in [-1, 0, 1]:
        if np.sum(y_test == cls) > 0:
            cls_mask_true = (y_test == cls)
            cls_mask_pred = (final_pred == cls)
            
            tp = np.sum(cls_mask_true & cls_mask_pred)
            recall = tp / np.sum(cls_mask_true)
            precision = tp / np.sum(cls_mask_pred) if np.sum(cls_mask_pred) > 0 else 0
            
            label = {-1: 'DOWN', 0: 'NEUTRAL', 1: 'UP'}.get(cls)
            print(f"\n{label}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
    
    unique, counts = np.unique(final_pred, return_counts=True)
    print(f"\nPrediction Distribution:")
    for cls, cnt in zip(unique, counts):
        label = {-1: 'DOWN', 0: 'NEUTRAL', 1: 'UP'}.get(cls, str(cls))
        print(f"  {label:>8s}: {cnt:>6,} ({cnt/len(y_test)*100:5.1f}%)")
    
    return model_up, model_down, final_pred, thresh_up, thresh_down


# --------------------------------------------------------------
# FINAL MODEL EGITIMI
# --------------------------------------------------------------
def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weights: np.ndarray,
    best_params: dict,
) -> xgb.XGBClassifier:
    """
    Optuna'nin buldugu en iyi parametrelerle final modeli egitir.
    """
    print(f"\n{'-'*60}")
    print("  FINAL MODEL EGITIMI (en iyi parametrelerle)")
    print(f"{'-'*60}")

    y_train_enc = encode_labels(y_train)
    y_val_enc   = encode_labels(y_val)

    params = {
        'objective':        'multi:softmax',
        'num_class':        3,
        'device':           'cuda',
        'tree_method':      'hist',
        'eval_metric':      'mlogloss',
        'random_state':     42,
        'verbosity':        0,
        'early_stopping_rounds': 50,  # Final'da daha sabirli
        **best_params,
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train_enc,
        eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)],
        sample_weight=sample_weights,
        verbose=False,
    )

    print(f"  Best iteration: {model.best_iteration}")
    return model


# --------------------------------------------------------------
# MODEL DEGERLENDIRME
# --------------------------------------------------------------
def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list,
    output_dir: Path,
) -> dict:
    """
    Test seti uzerinde model performansini degerlendirir.
    Raporlar ve grafikler olusturur.
    """
    print(f"\n{'-'*60}")
    print("  MODEL DEGERLENDIRME (Test Seti)")
    print(f"{'-'*60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred_enc = model.predict(X_test)
    y_pred = decode_labels(y_pred_enc)

    # -- Classification Report --
    target_names = ['DOWN (-1)', 'NEUTRAL (0)', 'UP (+1)']
    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        digits=4,
    )
    print(f"\n{report}")

    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"BTC XGBoost - Test Set Classification Report\n")
        f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        f.write(report)

    # -- Metrikler --
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)

    metrics = {
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'best_iteration': int(model.best_iteration),
        'timestamp': datetime.now().isoformat(),
    }

    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro F1    : {f1_macro:.4f}")
    print(f"  Weighted F1 : {f1_weighted:.4f}")

    # -- Confusion Matrix --
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
    _plot_confusion_matrix(cm, target_names, output_dir / 'confusion_matrix.png')

    # -- Feature Importance --
    _plot_feature_importance(model, feature_cols, output_dir / 'feature_importance.png')

    print(f"\n  Dosyalar kaydedildi -> {output_dir}")
    return metrics


def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    save_path: Path,
) -> None:
    """Confusion matrix heatmap'i cizer ve kaydeder."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel('Tahmin', fontsize=12)
    ax.set_ylabel('Gercek', fontsize=12)
    ax.set_title('Confusion Matrix - Test Set', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix -> {save_path}")


def _plot_feature_importance(
    model: xgb.XGBClassifier,
    feature_cols: list,
    save_path: Path,
    top_n: int = 20,
) -> None:
    """Top-N feature importance bar chart cizer."""
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1][:top_n]

    top_features = [feature_cols[i] for i in sorted_idx]
    top_scores   = importance[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_scores[::-1], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_title(f'Top-{top_n} Feature Importance', fontsize=14)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Feature importance -> {save_path}")


# --------------------------------------------------------------
# MODEL KAYDETME / YUKLEME
# --------------------------------------------------------------
def save_model(
    model_up: xgb.XGBClassifier,
    model_down: xgb.XGBClassifier,
    best_params: dict,
    thresh_up: float,
    thresh_down: float,
    output_dir: Path,
) -> None:
    """Modelleri ve threshold'ları diske kaydeder."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model_up_path = output_dir / 'model_up.json'
    model_up.save_model(str(model_up_path))
    
    model_down_path = output_dir / 'model_down.json'
    model_down.save_model(str(model_down_path))
    
    # Threshold'ları ve params'ları da kaydet
    meta_path = output_dir / 'meta.json'
    meta_data = {
        'thresh_up': float(thresh_up),
        'thresh_down': float(thresh_down),
        'best_params': best_params
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=4)

    print(f"  Modeller kaydedildi -> {output_dir}")


def load_model(model_dir: Path) -> xgb.XGBClassifier:
    """Kaydedilmis modeli diskten yukler."""
    model_path = model_dir / 'model.json'
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyasi bulunamadi: {model_path}")

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


# --------------------------------------------------------------
# TAM EGITIM PIPELINE
# --------------------------------------------------------------
def full_training_pipeline(
    timeframe: str,
    n_trials: int = 50,
    timeout: int = 1800,
    skip_tuning: bool = False,
) -> None:
    """
    Bir timeframe icin tam egitim pipeline'i:
    1. Veri yukleme
    2. Sample weight hesaplama
    3. Baseline model
    4. Optuna tuning (opsiyonel)
    5. Final model egitimi
    6. Test degerlendirme
    7. Kaydetme

    Parameters
    ----------
    timeframe    : '15m', '1h', '4h', '1d'
    n_trials     : Optuna trial sayisi
    timeout      : Tuning timeout (saniye)
    skip_tuning  : True ise Optuna atlanir, baseline parametreleriyle devam eder
    """
    print(f'\n{"="*70}')
    print(f'  XGBOOST EGITIMI - {timeframe.upper()} TIMEFRAME')
    print(f'{"="*70}')

    output_dir = MODEL_DIR / timeframe

    # 1. Veri yukleme
    print(f"\n[1/6] Veri yukleniyor...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_split_data(timeframe)

    # 2. Sample weights
    print(f"\n[2/6] Sinif agirliklari hesaplaniyor...")
    sample_weights = compute_sample_weights(y_train)

    # 3. Baseline
    print(f"\n[3/6] Baseline model egitiliyor...")
    baseline_model = train_baseline(X_train, y_train, X_val, y_val, sample_weights)

    if skip_tuning:
        # Tuning atla, baseline parametreleriyle devam et
        print(f"\n[4/6] Tuning atlandi (--skip-tuning)")
        best_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
    else:
        # 4. Optuna tuning
        print(f"\n[4/6] Hyperparameter tuning baslatiliyor...")
        best_params = run_optuna_tuning(
            X_train, y_train, X_val, y_val,
            sample_weights, n_trials=n_trials, timeout=timeout,
        )

    # 5. Binary Relevance Modelleri
    print(f"\n[5/6] Binary Relevance Modelleri Egitiliyor...")
    model_up, model_down, final_pred, thresh_up, thresh_down = train_binary_relevance_models_v2(
        X_train, y_train, X_val, y_val, X_test, y_test,
        sample_weights, best_params
    )

    # 6. Kaydetme
    print(f"\n[6/6] Modeller kaydediliyor...")
    save_model(model_up, model_down, best_params, thresh_up, thresh_down, output_dir)

    print(f'\n{"="*70}')
    print(f'  {timeframe.upper()} - EGITIM TAMAMLANDI')
    print(f'{"="*70}\n')
