import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
    matthews_corrcoef, precision_recall_curve, auc, average_precision_score
)
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

def clean_impossible_values(
    df: pd.DataFrame,
    drop: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Detects and optionally removes rows with impossible values in key numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing manufacturing process data.
    drop : bool, default=True
        If True, drops rows containing impossible values.
        If False, returns a copy of the original DataFrame with an 'ImpossibleValue' flag column.
    verbose : bool, default=True
        If True, prints diagnostic information about found values.
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (rows dropped or flagged).
    """
    # Define conditions for impossible values
    impossible_filters = {
        'Air temperature [K]': df['Air temperature [K]'] <= 0,
        'Process temperature [K]': df['Process temperature [K]'] <= 0,
        'Rotational speed [rpm]': df['Rotational speed [rpm]'] < 0,
        'Torque [Nm]': df['Torque [Nm]'] < 0,
        'Tool wear [min]': df['Tool wear [min]'] < 0
    }

    # Print summary of impossible values
    if verbose:
        for col, filt in impossible_filters.items():
            n = filt.sum()
            msg = f"Found {n} impossible rows in {col}." if n else f"No obvious impossible values in {col}."
            print(msg)

    # Combine all filters
    mask_any_impossible = np.zeros(len(df), dtype=bool)
    for filt in impossible_filters.values():
        mask_any_impossible |= filt

    # Handle according to 'drop' parameter
    if mask_any_impossible.any():
        if drop:
            df_cleaned = df.loc[~mask_any_impossible].copy()
            if verbose:
                print(f"Dropped {mask_any_impossible.sum()} rows with impossible values.")
        else:
            df_cleaned = df.copy()
            df_cleaned['ImpossibleValue'] = mask_any_impossible
            if verbose:
                print(f"Flagged {mask_any_impossible.sum()} rows with impossible values.")
    else:
        if verbose:
            print("No rows dropped or flagged (no impossible values found).")
        df_cleaned = df.copy()

    return df_cleaned

def remove_target_contradictions(
    df: pd.DataFrame,
    target_col: str = "Target",
    failure_col: str = "Failure Type",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Removes rows with contradictions between binary target and failure type columns.
    
    A contradiction is defined as:
        - Target == 1 and Failure Type == 'No Failure'
        - Target == 0 and Failure Type != 'No Failure'

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target and failure columns.
    target_col : str, default='Target'
        Name of the binary target column (0 = No Failure, 1 = Failure).
    failure_col : str, default='Failure Type'
        Name of the column describing the type of failure.
    verbose : bool, default=True
        If True, prints before/after contradiction counts.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with contradictions removed.
    """
    # Count contradictions before cleaning
    contradictions_before = df[
        ((df[target_col] == 1) & (df[failure_col] == 'No Failure')) |
        ((df[target_col] == 0) & (df[failure_col] != 'No Failure'))
    ].shape[0]
    
    if verbose:
        print(f"Contradictions before cleaning: {contradictions_before} rows.")

    # Create contradiction mask
    mask = ((df[target_col] == 1) & (df[failure_col] == 'No Failure')) | \
           ((df[target_col] == 0) & (df[failure_col] != 'No Failure'))

    # Remove contradictory rows
    df_cleaned = df.loc[~mask].copy()

    # Verify removal
    contradictions_after = df_cleaned[
        ((df_cleaned[target_col] == 1) & (df_cleaned[failure_col] == 'No Failure')) |
        ((df_cleaned[target_col] == 0) & (df_cleaned[failure_col] != 'No Failure'))
    ].shape[0]

    if verbose:
        print(f"Contradictions after cleaning: {contradictions_after} rows.")
        print(f"Removed {contradictions_before - contradictions_after} contradictory rows.\n")

    return df_cleaned

def clean_col_names(df):
    cols = df.columns
    new_cols = cols.str.replace(r'\[|\]|<', '', regex=True).str.replace(r'\s+', '_', regex=True)
    df.columns = new_cols
    return df

def binary_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_pr = auc(recall, precision)

    return dict(
        accuracy=acc, precision=prec, recall=rec, f1=f1,
        roc_auc=roc, mcc=mcc, confusion_matrix=cm, AUC_PR= auc_pr
    )

def tune_and_evaluate_stage1_models(X_train, y_train, X_test, y_test, f1_macro_scorer, n_iter=100, cv=3, random_state=42):
    """
    Tunes and evaluates RandomForest, LightGBM, and XGBoost models using RandomizedSearchCV.
    Prints and returns the best estimators and evaluation metrics for each model.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : array-like
        Training and testing datasets.
    f1_macro_scorer : callable
        Scoring function for RandomizedSearchCV.
    n_iter : int, default=100
        Number of parameter combinations sampled.
    cv : int, default=3
        Cross-validation folds.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing best estimators and evaluation metrics for each model.
    """

    results = {}

    # === RANDOM FOREST ===
    rf = RandomForestClassifier(random_state=random_state, class_weight='balanced', n_jobs=-1)
    rf_param_grid = {
        'n_estimators': [200, 250, 300],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['entropy']
    }

    rf_search = RandomizedSearchCV(
        rf, rf_param_grid, scoring=f1_macro_scorer, n_iter=n_iter,
        cv=cv, n_jobs=-1, verbose=2, random_state=random_state
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_

    y_pred_rf = best_rf.predict(X_test)
    y_probs_rf = best_rf.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else None
    auc_pr_rf = np.nan
    if y_probs_rf is not None:
        precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_probs_rf)
        auc_pr_rf = auc(recall_rf, precision_rf)

    print("\n--- Random Forest Results ---")
    print(f"Best Params: {rf_search.best_params_}")
    print(f"CV F1: {rf_search.best_score_:.4f}")
    print(classification_report(y_test, y_pred_rf, zero_division=0))
    print(f"AUC-PR: {auc_pr_rf:.4f}\n")

    results["RandomForest"] = {
        "best_estimator": best_rf,
        "best_params": rf_search.best_params_,
        "cv_f1": rf_search.best_score_,
        "test_f1_macro": f1_score(y_test, y_pred_rf, average='macro'),
        "auc_pr": auc_pr_rf
    }

    # === LIGHTGBM ===
    lgbm = LGBMClassifier(random_state=random_state, class_weight='balanced', n_jobs=-1)
    lgbm_param_grid = {
        'n_estimators': [250, 300, 350, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [-1, 10, 15],
        'num_leaves': [63, 75, 127],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_lambda': [0, 1, 2]
    }

    lgbm_search = RandomizedSearchCV(
        lgbm, lgbm_param_grid, scoring=f1_macro_scorer, n_iter=n_iter,
        cv=cv, n_jobs=-1, random_state=random_state
    )
    lgbm_search.fit(X_train, y_train)
    best_lgbm = lgbm_search.best_estimator_

    y_pred_lgbm = best_lgbm.predict(X_test)
    y_probs_lgbm = best_lgbm.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else None
    auc_pr_lgbm = np.nan
    if y_probs_lgbm is not None:
        precision_lgbm, recall_lgbm, _ = precision_recall_curve(y_test, y_probs_lgbm)
        auc_pr_lgbm = auc(recall_lgbm, precision_lgbm)

    print("\n--- LightGBM Results ---")
    print(f"Best Params: {lgbm_search.best_params_}")
    print(f"CV F1: {lgbm_search.best_score_:.4f}")
    print(classification_report(y_test, y_pred_lgbm, zero_division=0))
    print(f"AUC-PR: {auc_pr_lgbm:.4f}\n")

    results["LightGBM"] = {
        "best_estimator": best_lgbm,
        "best_params": lgbm_search.best_params_,
        "cv_f1": lgbm_search.best_score_,
        "test_f1_macro": f1_score(y_test, y_pred_lgbm, average='macro'),
        "auc_pr": auc_pr_lgbm
    }

    # === XGBOOST ===
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if len(np.unique(y_train)) == 2 else 1
    xgb = XGBClassifier(
        objective='binary:logistic' if len(np.unique(y_train)) == 2 else 'multi:softprob',
        eval_metric='logloss', random_state=random_state, n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    xgb_param_grid = {
        'n_estimators': [300, 350, 400],
        'learning_rate': [0.01, 0.05],
        'max_depth': [9, 10, 15],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.001, 0.01, 0.1],
        'reg_lambda': [0.1, 1]
    }

    xgb_search = RandomizedSearchCV(
        xgb, xgb_param_grid, scoring=f1_macro_scorer, n_iter=n_iter,
        cv=cv, verbose=2, n_jobs=-1, random_state=random_state
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    y_pred_xgb = best_xgb.predict(X_test)
    y_probs_xgb = best_xgb.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else None
    auc_pr_xgb = np.nan
    if y_probs_xgb is not None:
        precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_probs_xgb)
        auc_pr_xgb = auc(recall_xgb, precision_xgb)

    print("\n--- XGBoost Results ---")
    print(f"Best Params: {xgb_search.best_params_}")
    print(f"CV F1: {xgb_search.best_score_:.4f}")
    print(classification_report(y_test, y_pred_xgb, zero_division=0))
    print(f"AUC-PR: {auc_pr_xgb:.4f}\n")

    results["XGBoost"] = {
        "best_estimator": best_xgb,
        "best_params": xgb_search.best_params_,
        "cv_f1": xgb_search.best_score_,
        "test_f1_macro": f1_score(y_test, y_pred_xgb, average='macro'),
        "auc_pr": auc_pr_xgb
    }

    return results

def evaluate_thresholds(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    
    results = []
    for t, p, r, f1 in zip(thresholds, precisions[:-1], recalls[:-1], f1_scores[:-1]):
        results.append((t, p, r, f1))
    
    results = np.array(results)
    best_idx = np.argmax(results[:, 3])
    best_threshold = results[best_idx, 0]
    
    print(f"Best Threshold by F1: {best_threshold:.3f}")
    print(f"At that threshold — Precision: {results[best_idx,1]:.3f}, Recall: {results[best_idx,2]:.3f}, F1: {results[best_idx,3]:.3f}")
    return best_threshold, results

def evaluate_thresholds_r(y_true, y_proba, min_precision=0.9):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    
    results = []
    for t, p, r, f1 in zip(thresholds, precisions[:-1], recalls[:-1], f1_scores[:-1]):
        results.append((t, p, r, f1))
    results = np.array(results)
    
    # --- Best by F1 (existing logic)
    best_idx_f1 = np.argmax(results[:, 3])
    best_threshold_f1 = results[best_idx_f1, 0]
    
    # --- Best by Recall (under precision constraint)
    valid = results[results[:, 1] >= min_precision]
    if len(valid) > 0:
        best_idx_recall = np.argmax(valid[:, 2])
        best_threshold_recall = valid[best_idx_recall, 0]
        best_prec, best_rec, best_f1 = valid[best_idx_recall, 1:]
        print(f"\nBest Threshold for Recall (Precision ≥ {min_precision}): {best_threshold_recall:.3f}")
        print(f"Precision: {best_prec:.3f}, Recall: {best_rec:.3f}, F1: {best_f1:.3f}")
    else:
        best_threshold_recall = None
        print(f"\nNo threshold found with Precision ≥ {min_precision}")

    # --- F1-based results
    print(f"\nBest Threshold by F1: {best_threshold_f1:.3f}")
    print(f"At that threshold — Precision: {results[best_idx_f1,1]:.3f}, Recall: {results[best_idx_f1,2]:.3f}, F1: {results[best_idx_f1,3]:.3f}")

    return {
        "best_threshold_f1": best_threshold_f1,
        "best_threshold_recall": best_threshold_recall,
        "results": results
    }

def evaluate_metrics(y_true, y_pred, y_proba=None):
    # Base classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    roc_auc = np.nan
    auc_pr = np.nan

    # If probabilities are provided, handle both binary & multiclass cases
    if y_proba is not None:
        y_true_unique = np.unique(y_true)
        
        # Binary classification
        if len(y_true_unique) == 2:
            roc_auc = roc_auc_score(y_true, y_proba)
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            auc_pr = auc(recall, precision)
        
        # Multiclass classification (One-vs-Rest)
        else:
            y_true_bin = label_binarize(y_true, classes=y_true_unique)
            roc_auc = roc_auc_score(y_true_bin, y_proba, average='weighted', multi_class='ovr')

            auc_pr_scores = []
            for i in range(y_true_bin.shape[1]):
                prec_i, rec_i, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
                auc_pr_scores.append(auc(rec_i, prec_i))
            auc_pr = np.mean(auc_pr_scores)

    return dict(
        Accuracy=acc,
        Precision=prec,
        Recall=rec,
        F1=f1,
        MCC=mcc,
        ROC_AUC=roc_auc,
        Confusion_Matrix=cm,
        AUC_PR=auc_pr
    )

def train_stage2_models(X_train_smote, y_train_smote, X2_test, y2_test, le):
    """
    Trains and evaluates RandomForest, XGBoost, and LightGBM models using GridSearchCV.

    Parameters:
        X_train_smote (pd.DataFrame): Oversampled training features.
        y_train_smote (pd.Series): Oversampled training labels.
        X2_test (pd.DataFrame): Test set features.
        y2_test (pd.Series): Test set labels.
        le (LabelEncoder): Fitted label encoder for multi-class support.

    Returns:
        dict: Dictionary containing best models and their GridSearch results.
    """

    results = {}

    # --- Random Forest ---
    rf2 = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf2_params = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2]
    }

    rf2_grid = GridSearchCV(rf2, rf2_params, scoring='f1_macro', cv=3, verbose=1, n_jobs=-1)
    rf2_grid.fit(X_train_smote, y_train_smote)
    rf2_best = rf2_grid.best_estimator_

    print("\n--- Random Forest Results ---")
    print(f"Best Parameters: {rf2_grid.best_params_}")
    print(f"Best Cross-validation F1: {rf2_grid.best_score_:.4f}")

    y_pred_rf2 = rf2_best.predict(X2_test)
    print("\nClassification Report (Test Set):")
    print(classification_report(y2_test, y_pred_rf2))

    y2_test_binrf = label_binarize(y2_test, classes=rf2_best.classes_)
    y_probs_rf2 = rf2_best.predict_proba(X2_test)
    avg_precision_rf2 = average_precision_score(y2_test_binrf, y_probs_rf2, average='macro')
    print(f"Macro-average Precision-Recall AUC: {avg_precision_rf2:.4f}")

    results["rf"] = {
        "model": rf2_best,
        "params": rf2_grid.best_params_,
        "cv_score": rf2_grid.best_score_,
        "auc_pr": avg_precision_rf2
    }

    # --- XGBoost ---
    xgb2 = XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1,
        num_class=len(le.classes_)
    )
    xgb2_params = {
        'n_estimators': [150, 200, 250],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    }

    xgb2_grid = GridSearchCV(xgb2, xgb2_params, scoring='f1_macro', cv=3, verbose=1, n_jobs=-1)
    xgb2_grid.fit(X_train_smote, y_train_smote)
    xgb2_best = xgb2_grid.best_estimator_

    print("\n--- XGBoost Results ---")
    print(f"Best Parameters: {xgb2_grid.best_params_}")
    print(f"Best Cross-validation F1: {xgb2_grid.best_score_:.4f}")

    y_pred_xgb2 = xgb2_best.predict(X2_test)
    print("\nClassification Report (Test Set):")
    print(classification_report(y2_test, y_pred_xgb2, zero_division=0))

    y2_test_binxgb = label_binarize(y2_test, classes=xgb2_best.classes_)
    y_probs_xgb2 = xgb2_best.predict_proba(X2_test)
    avg_precision_xgb2 = average_precision_score(y2_test_binxgb, y_probs_xgb2, average='macro')
    print(f"Macro-average Precision-Recall AUC: {avg_precision_xgb2:.4f}")

    results["xgb"] = {
        "model": xgb2_best,
        "params": xgb2_grid.best_params_,
        "cv_score": xgb2_grid.best_score_,
        "auc_pr": avg_precision_xgb2
    }

    # --- LightGBM ---
    lgb2 = LGBMClassifier(
        objective='multiclass',
        random_state=42,
        n_jobs=-1,
        num_class=len(le.classes_)
    )
    lgb2_params = {
        'n_estimators': [100, 150, 200],
        'max_depth': [6, 7, 9, 10],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.6, 0.7],
        'num_leaves': [10, 20, 30],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    lgb2_grid = GridSearchCV(lgb2, lgb2_params, scoring='f1_macro', cv=3, verbose=1, n_jobs=-1)
    lgb2_grid.fit(X_train_smote, y_train_smote)
    lgb2_best = lgb2_grid.best_estimator_

    print("\n--- LightGBM Results ---")
    print(f"Best Parameters: {lgb2_grid.best_params_}")
    print(f"Best Cross-validation F1: {lgb2_grid.best_score_:.4f}")

    y_pred_lgbm2 = lgb2_best.predict(X2_test)
    print("\nClassification Report (Test Set):")
    print(classification_report(y2_test, y_pred_lgbm2))

    y2_test_binlgb = label_binarize(y2_test, classes=lgb2_best.classes_)
    y_probs_lgb2 = lgb2_best.predict_proba(X2_test)
    avg_precision_lgb2 = average_precision_score(y2_test_binlgb, y_probs_lgb2, average='macro')
    print(f"Macro-average Precision-Recall AUC: {avg_precision_lgb2:.4f}")

    results["lgbm"] = {
        "model": lgb2_best,
        "params": lgb2_grid.best_params_,
        "cv_score": lgb2_grid.best_score_,
        "auc_pr": avg_precision_lgb2
    }

    return results

