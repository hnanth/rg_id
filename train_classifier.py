#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 12:58:17 2026

@author: huyennhu

Optimized version with:
- PARALLEL MODEL TRAINING (train multiple models simultaneously)
- Full parallelization (n_jobs=-1)
- Adaptive class imbalance handling (SMOTE vs class_weight)
- Error fixes and robust error handling
- Vectorized operations for speed
- Progress tracking
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Feature lists
confidence_scores = ['global_plddt', 'mean_plddt_A', 'mean_plddt_B', 'ptm',
       'iptm', 'combined_score', 'ipSAE_max', 'pDockQ_max', 'pDockQ2_max',
       'LIS_max', 'ipsae_A_B', 'pdockq2_A_B', 'lis_A_B', 'ipsae_B_A',
       'pdockq2_B_A', 'lis_B_A']
pyrosetta_scores = ['interface_dG', 'interface_SASA',
       'interface_dG_SASA_ratio', 'shape_complementarity',
       'num_interface_residues', 'buried_unsatisfied_hbonds', 'packstat',
       'binder_dG', 'binder_SASA']
electrostatics = ['idp_core_charged',
       'receptor_core_charged', 'idp_interface_charged',
       'receptor_interface_charged', 'idp_rim_charged', 'receptor_rim_charged',
       'idp_surface_charged', 'receptor_surface_charged',
       'idp_core_hydropathy', 'receptor_core_hydropathy',
       'idp_interface_hydropathy', 'receptor_interface_hydropathy',
       'idp_rim_hydropathy', 'receptor_rim_hydropathy',
       'idp_surface_hydropathy', 'receptor_surface_hydropathy',
       'idp_core_hydrophobic', 'receptor_core_hydrophobic',
       'idp_interface_hydrophobic', 'receptor_interface_hydrophobic',
       'idp_rim_hydrophobic', 'receptor_rim_hydrophobic',
       'idp_surface_hydrophobic', 'receptor_surface_hydrophobic',
       'idp_core_polar', 'receptor_core_polar', 'idp_interface_polar',
       'receptor_interface_polar', 'idp_rim_polar', 'receptor_rim_polar',
       'idp_surface_polar', 'receptor_surface_polar', 'idp_core_small',
       'receptor_core_small', 'idp_interface_small',
       'receptor_interface_small', 'idp_rim_small', 'receptor_rim_small',
       'idp_surface_small', 'receptor_surface_small']
all_features = confidence_scores + pyrosetta_scores + electrostatics


def get_balancing_strategy(y_train, section):
    """
    Determine best balancing strategy based on class imbalance ratio.
    
    Returns:
        strategy: 'none', 'class_weight', or 'smote'
        ratio: imbalance ratio
        reason: explanation
    """
    class_counts = y_train.value_counts()
    
    if len(class_counts) < 2:
        return 'none', None, 'single_class'
    
    ratio = class_counts.max() / class_counts.min()
    minority_size = class_counts.min()
    
    # Severe imbalance (>10:1): class_weight already applied to models
    if ratio > 10:
        return 'none', ratio, 'severe_imbalance_use_class_weight'
    
    # Moderate imbalance (2:1 to 10:1): add SMOTE on top of class_weight
    elif ratio >= 2:
        if minority_size >= 6:
            return 'smote', ratio, 'moderate_imbalance'
        else:
            return 'none', ratio, 'insufficient_samples_for_smote'
    
    # Balanced (<2:1): class_weight only (already in models)
    return 'none', ratio, 'balanced'


def compute_sample_weights(y):
    """Compute sample weights for GradientBoosting"""
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, class_weights))
    return np.array([weight_dict[yi] for yi in y])


def filter_dataset(df, criteria_name, threshold):
    """
    Filter dataset based on criteria.
    Optimized: uses vectorized operations instead of apply/lambda.
    """
    filtered = df.copy()
    
    # Determine column name
    if "contacts" in criteria_name:
        col = 'interacting_residues'
    elif "% interacting residues" in criteria_name:
        col = 'perc_interacting_residues'
    elif "% overlap interface" in criteria_name:
        col = 'perc_overlap_interface'
    elif "% overlap fragment" in criteria_name:
        col = 'perc_overlap_fragment'
    else:
        col = 'interacting_residues'  # default
    
    # Vectorized filtering (much faster than apply/lambda)
    if col == "perc_overlap_fragment":
        filtered["interacting_complex"] = (
            (filtered[f"idp_{col}"] >= threshold) & 
            (filtered["receptor_perc_overlap_domain"] >= threshold)
        )
    else:
        filtered["interacting_complex"] = (
            (filtered[f"idp_{col}"] >= threshold) & 
            (filtered[f"receptor_{col}"] >= threshold)
        )
    
    return filtered


def train_single_model(name, base_model, X_train, y_train, X_test, y_test, 
                       strategy, section, criteria_name, class_counts, section_df):
    """
    Train a single model - designed to be run in parallel.
    
    This function is completely self-contained so it can be safely parallelized.
    """
    try:
        # Create pipeline based on strategy
        if strategy == 'smote':
            k_neighbors = min(5, y_train.value_counts().min() - 1)
            pipe = IMBPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42, k_neighbors=k_neighbors)),
                ("classifier", base_model)
            ])
            sample_weight = None
        else:
            # For GradientBoosting, add sample_weight
            if isinstance(base_model, GradientBoostingClassifier):
                sample_weight = compute_sample_weights(y_train)
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", base_model)
                ])
            else:
                sample_weight = None
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", base_model)
                ])
        
        # Cross-validation (3 folds for speed)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Handle CV for GradientBoosting with sample weights
        if sample_weight is not None:
            # Manual CV for GradientBoosting
            cv_scores = []
            for train_idx, val_idx in cv.split(X_train, y_train):
                from sklearn.base import clone
                cv_pipe = clone(pipe)
                
                X_cv_train = X_train.iloc[train_idx]
                y_cv_train = y_train.iloc[train_idx]
                sw_train = compute_sample_weights(y_cv_train)
                
                cv_pipe.named_steps['scaler'].fit(X_cv_train)
                X_cv_train_scaled = cv_pipe.named_steps['scaler'].transform(X_cv_train)
                cv_pipe.named_steps['classifier'].fit(X_cv_train_scaled, y_cv_train, 
                                                      sample_weight=sw_train)
                
                X_cv_val = X_train.iloc[val_idx]
                y_cv_val = y_train.iloc[val_idx]
                X_cv_val_scaled = cv_pipe.named_steps['scaler'].transform(X_cv_val)
                y_cv_pred_proba = cv_pipe.named_steps['classifier'].predict_proba(X_cv_val_scaled)[:, 1]
                
                cv_scores.append(average_precision_score(y_cv_val, y_cv_pred_proba))
            cv_scores = np.array(cv_scores)
        else:
            # For models that support n_jobs, we set it to 1 here since we're already parallelizing models
            cv_scores = cross_val_score(
                pipe, X_train, y_train,
                cv=cv, 
                scoring="average_precision",
                n_jobs=1  # Don't parallelize within each model since models are parallel
            )
        
        # Train on full training set
        if sample_weight is not None:
            pipe.named_steps['scaler'].fit(X_train)
            X_train_scaled = pipe.named_steps['scaler'].transform(X_train)
            pipe.named_steps['classifier'].fit(X_train_scaled, y_train, 
                                               sample_weight=sample_weight)
        else:
            pipe.fit(X_train, y_train)
        
        # Predict
        y_pred = pipe.predict(X_test)
        y_pred_proba = pipe.predict_proba(X_test)[:, 1]
        
        # Evaluate (with error handling)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        ap = average_precision_score(y_test, y_pred_proba)
        
        try:
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            accuracy = np.nan
            auc = np.nan
        
        # Get imbalance ratio
        ratio = class_counts.max() / class_counts.min() if len(class_counts) >= 2 else np.nan
        
        # Return results as dictionary
        result = {
            'Model': name,
            'Section': section,
            'Criteria': criteria_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc,
            'Average Precision': ap,
            'CV PR-AUC scores': cv_scores,
            'Mean CV PR-AUC': np.mean(cv_scores),
            'Std CV PR-AUC': np.std(cv_scores),
            'Balancing Strategy': strategy,
            'Imbalance Ratio': ratio,
            'True Percentage': class_counts.get(True, 0) / len(section_df),
            'Train Size': len(X_train),
            'Test Size': len(X_test)
        }
        
        print(f"  ✓ {name}: AP={ap:.4f}, F1={f1:.4f}")
        return result
        
    except Exception as e:
        print(f"  ✗ {name}: ERROR - {str(e)}")
        return None


# Load dataset once
print("Loading dataset...")
df = pd.read_csv("uniprot_receptor_domain_idp_full_dataset.tsv", sep='\t')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

# Define sections and criteria
sections = ["FD", "AFminD (30)", "AFminD (50)", "AFminD (100)"]
criteria = {
    "Min. 5 contacts": 5,
    "40% interacting residues": 0.4,
    "50% interacting residues": 0.5,
    "80% interacting residues": 0.8,
    "40% overlap interface": 0.4,
    "50% overlap interface": 0.5,
    "80% overlap interface": 0.8,
    "40% overlap fragment": 0.4,
    "50% overlap fragment": 0.5,
    "80% overlap fragment": 0.8
}

# Base models with class_weight
# Note: n_jobs is set to 1 here since we parallelize at model level
base_models = {
    'Logistic Regression': LogisticRegression(
        class_weight="balanced", 
        random_state=42, 
        max_iter=1000,
        n_jobs=1  # Set to 1, will parallelize models instead
    ),
    'Random Forest': RandomForestClassifier(
        class_weight="balanced", 
        n_estimators=100, 
        random_state=42,
        n_jobs=1,  # Set to 1, will parallelize models instead
        max_depth=10
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=5,
        learning_rate=0.1
    ),
    'SVM': SVC(
        class_weight="balanced", 
        kernel='rbf', 
        probability=True, 
        random_state=42,
        cache_size=1000
    )
}

# Training loop
results = []
total_combinations = len(criteria) * len(sections)
current_combination = 0

# Determine number of parallel jobs
# Use all CPU cores for training models in parallel
N_JOBS_MODELS = -1  # -1 means use all available cores

print(f"Parallel model training enabled: Using all CPU cores")
print(f"Total combinations to process: {total_combinations}")
print("=" * 70)

for criteria_name, threshold in criteria.items():
    print(f"\n{'='*70}")
    print(f"CRITERIA: {criteria_name}")
    print('='*70)
    
    # Filter dataset (vectorized - faster than original)
    filtered = filter_dataset(df, criteria_name, threshold)
    
    for section in sections:
        current_combination += 1
        print(f"\n{'-'*70}")
        print(f"[{current_combination}/{total_combinations}] Section: {section}")
        print('-'*70)
        
        # Filter by section
        section_df = filtered[filtered['section'] == section]
        
        # Check if we have enough data
        if len(section_df) < 10:
            print(f"⚠ WARNING: Only {len(section_df)} samples. Skipping...")
            continue
        
        X = section_df[all_features]
        y = section_df["interacting_complex"]
        
        # Check class distribution
        class_counts = y.value_counts()
        print(f"\nClass distribution:")
        print(f"  True: {class_counts.get(True, 0)} ({class_counts.get(True, 0)/len(y)*100:.1f}%)")
        print(f"  False: {class_counts.get(False, 0)} ({class_counts.get(False, 0)/len(y)*100:.1f}%)")
        
        # Skip if only one class
        if len(class_counts) < 2:
            print(f"⚠ WARNING: Only one class present. Skipping...")
            continue
        
        # Split into training and testing sets
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        except ValueError as e:
            print(f"⚠ ERROR in train_test_split: {e}")
            continue
        
        print(f"\nData split:")
        print(f"  Training: {X_train.shape[0]} samples")
        print(f"  Testing: {X_test.shape[0]} samples")
        
        # Determine balancing strategy
        strategy, ratio, reason = get_balancing_strategy(y_train, section)
        print(f"\nBalancing Strategy:")
        print(f"  Imbalance ratio: {ratio:.2f}:1" if ratio else "  N/A")
        print(f"  Strategy: {strategy.upper()} (on top of class_weight)")
        print(f"  Reason: {reason}")
        
        # PARALLEL MODEL TRAINING
        print(f"\nTraining {len(base_models)} models in parallel...")
        
        # Train all models in parallel using joblib
        model_results = Parallel(n_jobs=N_JOBS_MODELS, verbose=0)(
            delayed(train_single_model)(
                name, base_model, 
                X_train, y_train, X_test, y_test,
                strategy, section, criteria_name, 
                class_counts, section_df
            )
            for name, base_model in base_models.items()
        )
        
        # Add non-None results to results list
        for result in model_results:
            if result is not None:
                results.append(result)

# Save and display results
if results:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['Section', 'Criteria', 'Average Precision'], 
                                        ascending=[True, True, False])
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON - ALL RESULTS")
    print('='*70)
    
    # Display key columns
    display_cols = ['Model', 'Section', 'Criteria', 'F1-Score', 'Average Precision', 
                    'Balancing Strategy', 'Imbalance Ratio']
    print(results_df[display_cols].to_string(index=False))
    
    # Save to file
    output_file = "classifier_training_parallel_models.tsv"
    results_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY BY MODEL")
    print('='*70)
    summary = results_df.groupby('Model').agg({
        'F1-Score': ['mean', 'std', 'count'],
        'Average Precision': ['mean', 'std']
    }).round(4)
    print(summary)
    
    print(f"\n{'='*70}")
    print("SUMMARY BY BALANCING STRATEGY")
    print('='*70)
    strategy_summary = results_df.groupby('Balancing Strategy').agg({
        'F1-Score': ['mean', 'std', 'count'],
        'Average Precision': ['mean', 'std']
    }).round(4)
    print(strategy_summary)
    
    print(f"\n{'='*70}")
    print("SUMMARY BY SECTION")
    print('='*70)
    section_summary = results_df.groupby('Section').agg({
        'F1-Score': ['mean', 'std'],
        'Average Precision': ['mean', 'std'],
        'Imbalance Ratio': 'mean'
    }).round(4)
    print(section_summary)
else:
    print("\n⚠ No results generated. Check your data and criteria.")