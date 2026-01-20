"""
GENiAL Phase 2: Machine Learning Classification
Predicting cluster membership from EEG features using Random Forest, SVM, and MLP

Author: Auto-generated
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc, cohen_kappa_score
)

# Feature Selection
# RFECV = Recursive feature elimination with cross-validation
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, RFECV,
    SelectFromModel
)

# Explainable AI
import shap
from sklearn.inspection import permutation_importance

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_PATH = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/Final/merged_clustered_EEG_features_global_RSRio_DEC_16_2025_logtransformed.csv"
OUTPUT_DIR = Path("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/CLEAN/DATA/Outputs/ML_Results")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# EEG features to use
EEG_FEATURES = [
    "hurst_2s", "fooof_offset_2s", "fooof_exponent_2s", 
    "pow_per_delta_2s", "pow_per_theta_2s", "pow_per_alpha_2s",
    "pow_per_beta_2s", "pow_per_gamma_2s", "pow_per_low_gamma_2s", "pow_per_high_gamma_2s",
    "higuchi_fd_5s", "katz_fd_5s", "samp_entropy_5s", "CI_5s", "CI_lowscale_5s",
    "CI_highscale_5s", "log_pow_delta_2s", "log_pow_theta_2s", "log_pow_alpha_2s",
    "log_pow_beta_2s", "log_pow_gamma_2s", "log_pow_low_gamma_2s", "log_pow_high_gamma_2s"
]

TARGET_COLUMN = "cluster"

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(data_path, eeg_features, target_column):
    """
    Load data and prepare features and target.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
    eeg_features : list
        List of EEG feature column names
    target_column : str
        Name of the target column (cluster)
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    df : pd.DataFrame
        Full dataframe
    """
    print("="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"\nLoaded data shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Check for target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Check for EEG features
    missing_features = [f for f in eeg_features if f not in df.columns]
    if missing_features:
        print(f"WARNING: Missing features: {missing_features}")
        eeg_features = [f for f in eeg_features if f in df.columns]
    
    print(f"\nUsing {len(eeg_features)} EEG features")
    
    # Extract features and target
    X = df[eeg_features].copy()
    y = df[target_column].copy()
    
    # Remove rows with missing values in features or target
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].copy()
    y = y[mask].copy()
    
    print(f"\nAfter removing missing values:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target distribution:\n{y.value_counts().sort_index()}")
    
    # Check for sufficient samples per class
    class_counts = y.value_counts()
    min_samples = class_counts.min()
    print(f"\nMinimum samples per class: {min_samples}")
    
    if min_samples < 5:
        print("WARNING: Some classes have very few samples (<5)")
    
    return X, y, df[mask].copy()


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def perform_feature_selection(X, y, method='mutual_info', n_features=None, cv=5):
    """
    Perform feature selection using various methods.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    method : str
        Method to use: 'mutual_info', 'f_test', 'rfe', 'rf_importance'
    n_features : int or None
        Number of features to select. If None, uses cross-validation
    cv : int
        Number of CV folds for RFECV
    
    Returns:
    --------
    selected_features : list
        List of selected feature names
    selector : object
        The fitted selector object
    """
    print("\n" + "="*80)
    print(f"FEATURE SELECTION: {method.upper()}")
    print("="*80)
    
    if n_features is None:
        n_features = min(15, X.shape[1] // 2)  # Default to half features or 15, whichever is smaller
    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Print feature scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        print("\nTop features by mutual information:")
        print(scores.head(20).to_string(index=False))
        
    elif method == 'f_test':
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Print feature scores
        scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        print("\nTop features by F-test:")
        print(scores.head(20).to_string(index=False))
        
    elif method == 'rfe':
        # Use Random Forest as base estimator for RFE
        base_estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        selector = RFE(estimator=base_estimator, n_features_to_select=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Print feature rankings
        rankings = pd.DataFrame({
            'feature': X.columns,
            'ranking': selector.ranking_
        }).sort_values('ranking')
        print("\nFeature rankings (1 = selected):")
        print(rankings.head(20).to_string(index=False))
        
    elif method == 'rf_importance':
        # Use Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X, y)
        
        # Select features based on importance threshold
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top n_features
        selected_features = importances.head(n_features)['feature'].tolist()
        selector = SelectFromModel(rf, prefit=True, max_features=n_features)
        
        print("\nTop features by Random Forest importance:")
        print(importances.head(20).to_string(index=False))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"\nSelected {len(selected_features)} features:")
    print(selected_features)
    
    return selected_features, selector


# ============================================================================
# MODEL OPTIMIZATION
# ============================================================================

def optimize_random_forest(X_train, y_train, cv=5, n_iter=50):
    """Optimize Random Forest hyperparameters."""
    print("\n" + "="*80)
    print("OPTIMIZING RANDOM FOREST")
    print("="*80)
    
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    # Use RandomizedSearchCV for faster search
    search = RandomizedSearchCV(
        rf, param_grid, n_iter=n_iter, cv=cv, 
        scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")
    
    return search.best_estimator_


def optimize_svm(X_train, y_train, cv=5, n_iter=30):
    """Optimize SVM hyperparameters."""
    print("\n" + "="*80)
    print("OPTIMIZING SVM")
    print("="*80)
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    svm = SVC(random_state=RANDOM_STATE, probability=True)
    
    search = RandomizedSearchCV(
        svm, param_grid, n_iter=n_iter, cv=cv,
        scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE,
        verbose=1
    )
    
    search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")
    
    return search.best_estimator_, scaler


def optimize_mlp(X_train, y_train, cv=5, n_iter=30):
    """Optimize MLP hyperparameters."""
    print("\n" + "="*80)
    print("OPTIMIZING MLP")
    print("="*80)
    
    # Scale features for MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (150,), (50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500, 1000]
    }
    
    mlp = MLPClassifier(random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.1)
    
    search = RandomizedSearchCV(
        mlp, param_grid, n_iter=n_iter, cv=cv,
        scoring='f1_macro', n_jobs=-1, random_state=RANDOM_STATE,
        verbose=1
    )
    
    search.fit(X_train_scaled, y_train)
    
    print(f"\nBest parameters: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")
    
    return search.best_estimator_, scaler


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name, scaler=None):
    """
    Evaluate a model and return metrics.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame or np.array
        Test features
    y_test : pd.Series or np.array
        Test labels
    model_name : str
        Name of the model
    scaler : sklearn scaler or None
        Scaler to apply to features if needed
    
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # Multi-class ROC AUC (one-vs-rest)
    try:
        if len(np.unique(y_test)) > 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    except:
        roc_auc = None
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'kappa': kappa,
        'roc_auc': roc_auc
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Kappa:     {kappa:.4f}")
    if roc_auc:
        print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics, y_pred, y_pred_proba


# ============================================================================
# EXPLAINABLE AI
# ============================================================================

def explain_with_shap(model, X_train, X_test, model_name, scaler=None, max_samples=100):
    """
    Generate SHAP explanations for the model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_train : pd.DataFrame or np.array
        Training features (for background)
    X_test : pd.DataFrame or np.array
        Test features (for explanation)
    model_name : str
        Name of the model
    scaler : sklearn scaler or None
        Scaler if needed
    max_samples : int
        Maximum samples to use for SHAP (for speed)
    """
    print(f"\n{'='*80}")
    print(f"SHAP EXPLANATIONS: {model_name}")
    print("="*80)
    
    # Preserve feature names
    if isinstance(X_test, pd.DataFrame):
        feature_names = X_test.columns.tolist()
    else:
        feature_names = None
    
    if scaler is not None:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_test_scaled = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    
    # Limit samples for speed
    if len(X_train_scaled) > max_samples:
        background = X_train_scaled[:max_samples]
    else:
        background = X_train_scaled
    
    if len(X_test_scaled) > max_samples:
        explain_data = X_test_scaled[:max_samples]
    else:
        explain_data = X_test_scaled
    
    try:
        # Create SHAP explainer
        if isinstance(model, RandomForestClassifier):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, background)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(explain_data)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            # Multi-class: use first class for summary plot
            shap.summary_plot(shap_values[0], explain_data, 
                            feature_names=feature_names,
                            show=False, max_display=20)
            plt.title(f"SHAP Summary Plot - {model_name} (Class 0)")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"shap_summary_{model_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            shap.summary_plot(shap_values, explain_data,
                            feature_names=feature_names,
                            show=False, max_display=20)
            plt.title(f"SHAP Summary Plot - {model_name}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"shap_summary_{model_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"SHAP plots saved to {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")
        print("Skipping SHAP analysis...")


def plot_feature_importance(models, model_names, feature_names, scalers=None):
    """
    Plot feature importance for all models.
    
    Parameters:
    -----------
    models : list
        List of trained models
    model_names : list
        List of model names
    feature_names : list
        List of feature names
    scalers : list or None
        List of scalers (if any)
    """
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
    if n_models == 1:
        axes = [axes]
    
    models_to_plot = []
    names_to_plot = []
    axes_to_use = []
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        # Get feature importance
        importances = None
        
        if hasattr(model, 'feature_importances_'):
            # Random Forest
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # SVM or MLP - use absolute coefficients
            if len(model.coef_.shape) > 1:
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                importances = np.abs(model.coef_)
        
        if importances is not None:
            models_to_plot.append((importances, name))
            axes_to_use.append(axes[idx])
    
    # Adjust figure if we have fewer models
    if len(models_to_plot) < n_models:
        fig.delaxes(axes[len(models_to_plot)])
    
    for (importances, name), ax in zip(models_to_plot, axes_to_use):
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        top_features = importance_df.head(20)
        ax.barh(range(len(top_features)), top_features['importance'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Importance')
        ax.set_title(f'{name} - Top 20 Features')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {OUTPUT_DIR}")


def plot_confusion_matrices(models, model_names, X_test, y_test, scalers=None):
    """Plot confusion matrices for all models."""
    print("\n" + "="*80)
    print("CONFUSION MATRICES")
    print("="*80)
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, (model, name) in enumerate(zip(models, model_names)):
        ax = axes[idx]
        
        if scalers and scalers[idx] is not None:
            X_test_scaled = scalers[idx].transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=sorted(y_test.unique()),
                   yticklabels=sorted(y_test.unique()))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{name}')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to {OUTPUT_DIR}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main analysis pipeline."""
    print("\n" + "="*80)
    print(" " * 25 + "GENiAL PHASE 2: MACHINE LEARNING")
    print("="*80 + "\n")
    
    # 1. Load and preprocess data
    X, y, df = load_and_preprocess_data(DATA_PATH, EEG_FEATURES, TARGET_COLUMN)
    
    # 2. Split data
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 3. Feature selection
    print("\n" + "="*80)
    print("FEATURE SELECTION")
    print("="*80)
    print("Trying multiple feature selection methods...")
    
    # Try different methods and select the best
    methods = ['mutual_info', 'f_test', 'rf_importance']
    best_features = None
    best_score = -1
    
    for method in methods:
        try:
            selected_features, selector = perform_feature_selection(
                X_train, y_train, method=method, n_features=15
            )
            
            # Evaluate with a simple RF
            X_train_selected = X_train[selected_features]
            rf_temp = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
            scores = cross_val_score(rf_temp, X_train_selected, y_train, cv=5, scoring='f1_macro')
            mean_score = scores.mean()
            
            print(f"{method} CV F1-score: {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_features = selected_features
                best_selector = selector
        except Exception as e:
            print(f"Error with {method}: {e}")
            continue
    
    print(f"\nBest feature selection method selected {len(best_features)} features")
    print(f"Selected features: {best_features}")
    
    # Use selected features
    X_train_selected = X_train[best_features]
    X_test_selected = X_test[best_features]
    
    # 4. Optimize models
    print("\n" + "="*80)
    print("MODEL OPTIMIZATION")
    print("="*80)
    
    # Random Forest
    rf_best = optimize_random_forest(X_train_selected, y_train, cv=5, n_iter=50)
    
    # SVM
    svm_best, svm_scaler = optimize_svm(X_train_selected, y_train, cv=5, n_iter=30)
    
    # MLP
    mlp_best, mlp_scaler = optimize_mlp(X_train_selected, y_train, cv=5, n_iter=30)
    
    # 5. Evaluate models
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    all_metrics = []
    all_predictions = {}
    all_probabilities = {}
    
    # Random Forest
    rf_metrics, rf_pred, rf_proba = evaluate_model(
        rf_best, X_test_selected, y_test, "Random Forest"
    )
    all_metrics.append(rf_metrics)
    all_predictions['Random Forest'] = rf_pred
    all_probabilities['Random Forest'] = rf_proba
    
    # SVM
    svm_metrics, svm_pred, svm_proba = evaluate_model(
        svm_best, X_test_selected, y_test, "SVM", scaler=svm_scaler
    )
    all_metrics.append(svm_metrics)
    all_predictions['SVM'] = svm_pred
    all_probabilities['SVM'] = svm_proba
    
    # MLP
    mlp_metrics, mlp_pred, mlp_proba = evaluate_model(
        mlp_best, X_test_selected, y_test, "MLP", scaler=mlp_scaler
    )
    all_metrics.append(mlp_metrics)
    all_predictions['MLP'] = mlp_pred
    all_probabilities['MLP'] = mlp_proba
    
    # 6. Compare models
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    metrics_df = pd.DataFrame(all_metrics)
    print("\nModel Comparison:")
    print(metrics_df.to_string(index=False))
    
    # Save metrics
    metrics_df.to_csv(OUTPUT_DIR / 'model_comparison_metrics.csv', index=False)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'kappa', 'roc_auc']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        if metric in metrics_df.columns:
            metrics_df.plot(x='model', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Score')
            ax.set_xticklabels(metrics_df['model'], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Explainable AI
    print("\n" + "="*80)
    print("EXPLAINABLE AI")
    print("="*80)
    
    # SHAP explanations
    try:
        explain_with_shap(rf_best, X_train_selected, X_test_selected, "Random Forest")
        explain_with_shap(svm_best, X_train_selected, X_test_selected, "SVM", scaler=svm_scaler)
        explain_with_shap(mlp_best, X_train_selected, X_test_selected, "MLP", scaler=mlp_scaler)
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
    
    # Feature importance plots
    plot_feature_importance(
        [rf_best, svm_best, mlp_best],
        ["Random Forest", "SVM", "MLP"],
        best_features,
        scalers=[None, svm_scaler, mlp_scaler]
    )
    
    # Confusion matrices
    plot_confusion_matrices(
        [rf_best, svm_best, mlp_best],
        ["Random Forest", "SVM", "MLP"],
        X_test_selected, y_test,
        scalers=[None, svm_scaler, mlp_scaler]
    )
    
    # Permutation importance
    print("\nCalculating permutation importance...")
    for model, name, scaler in zip([rf_best, svm_best, mlp_best], 
                                   ["Random Forest", "SVM", "MLP"],
                                   [None, svm_scaler, mlp_scaler]):
        try:
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test_selected)
            else:
                X_test_scaled = X_test_selected.values if isinstance(X_test_selected, pd.DataFrame) else X_test_selected
            
            perm_importance = permutation_importance(
                model, X_test_scaled, y_test, n_repeats=10, 
                random_state=RANDOM_STATE, n_jobs=-1, scoring='f1_macro'
            )
            
            perm_df = pd.DataFrame({
                'feature': best_features,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            print(f"\n{name} - Permutation Importance (Top 10):")
            print(perm_df.head(10).to_string(index=False))
            
            perm_df.to_csv(OUTPUT_DIR / f'permutation_importance_{name.lower().replace(" ", "_")}.csv', index=False)
            
        except Exception as e:
            print(f"Error calculating permutation importance for {name}: {e}")
    
    # 8. Summary
    print("\n" + "="*80)
    print(" " * 25 + "ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nOutput files:")
    print("  - model_comparison_metrics.csv")
    print("  - model_comparison.png")
    print("  - feature_importance_comparison.png")
    print("  - confusion_matrices.png")
    print("  - shap_summary_*.png")
    print("  - permutation_importance_*.csv")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
