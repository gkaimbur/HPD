#!/usr/bin/env python3
"""
OPTIMIZED Maternal Health Risk Classification Model
Improvements:
1. Combine real data (Maternal_Risk.csv 1014) + synthetic data (2500)
2. SMOTE for class imbalance
3. Hyperparameter tuning
4. Larger validation sets
5. Advanced evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    f1_score, accuracy_score, cohen_kappa_score, matthews_corrcoef
)
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Try SMOTE for class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("WARNING: imbalanced-learn not installed. Install with: pip install imbalanced-learn")

os_path = '/memories/session/'

print("=" * 80)
print("OPTIMIZED MATERNAL HEALTH RISK CLASSIFICATION MODEL")
print("=" * 80)

# ============================================================================
# 1. LOAD REAL DATA + SYNTHETIC DATA
# ============================================================================
print("\n[1] LOADING DATA...")

# Load real maternal risk data
df_real = pd.read_csv('Maternal_Risk.csv')
print(f"[OK] Real data shape: {df_real.shape}")

# Load synthetic patient data (only relevant columns)
df_synthetic = pd.read_csv('synthetic_patients_2500.csv')
print(f"[OK] Synthetic data shape: {df_synthetic.shape}")

# Map synthetic data to risk levels based on risk score
# Use quartiles to classify into 3 risk levels
synthetic_risk_quantiles = df_synthetic['risk'].quantile([0.33, 0.67]).values
print(f"\nRisk quantiles for classification: {synthetic_risk_quantiles}")

def map_risk_score_to_level(risk_score):
    """Map continuous risk score to categorical risk level"""
    if risk_score < synthetic_risk_quantiles[0]:
        return 'low risk'
    elif risk_score < synthetic_risk_quantiles[1]:
        return 'mid risk'
    else:
        return 'high risk'

df_synthetic['RiskLevel'] = df_synthetic['risk'].apply(map_risk_score_to_level)

# Create features from synthetic data (match real data columns)
# Map synthetic columns to real data columns
df_synthetic_mapped = pd.DataFrame({
    'Age': (2026 - df_synthetic['dob'].str[:4].astype(int)),
    'SystolicBP': df_synthetic['sbp'],
    'DiastolicBP': df_synthetic['dbp'],
    'BS': df_synthetic['blood_sugar'] / 10,  # Scale to similar range
    'BodyTemp': 98 + (np.random.normal(0, 0.5, len(df_synthetic))),  # Add variation
    'HeartRate': 70 + df_synthetic['bmi'].astype(int),  # Estimate from BMI
    'RiskLevel': df_synthetic['RiskLevel']
})

print(f"\n[OK] Synthetic data mapped: {df_synthetic_mapped.shape}")

# Combine real + synthetic data
df_combined = pd.concat([df_real, df_synthetic_mapped], ignore_index=True)
print(f"[OK] Combined data shape: {df_combined.shape}")
print(f"\nRisk Level Distribution (Combined):")
print(df_combined['RiskLevel'].value_counts())

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n[2] PREPROCESSING DATA...")

# Encode target variable
le = LabelEncoder()
df_combined['RiskLevel_encoded'] = le.fit_transform(df_combined['RiskLevel'])

# Separate features and target
X = df_combined.drop(['RiskLevel', 'RiskLevel_encoded'], axis=1)
y = df_combined['RiskLevel_encoded']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[OK] Training set: {X_train.shape}")
print(f"[OK] Test set: {X_test.shape}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to handle class imbalance
if SMOTE_AVAILABLE:
    print(f"\n[OK] Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_scaled_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"    After SMOTE - Training set: {X_train_scaled_balanced.shape}")
    print(f"    Class distribution: {np.bincount(y_train_balanced)}")
else:
    X_train_scaled_balanced = X_train_scaled
    y_train_balanced = y_train
    print(f"[WARNING] SMOTE not available - using original class distribution")

# ============================================================================
# 3. HYPERPARAMETER TUNING WITH GRID SEARCH
# ============================================================================
print("\n[3] HYPERPARAMETER TUNING...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

base_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

# Use smaller grid for faster tuning
param_grid_small = {
    'n_estimators': [100, 150],
    'max_depth': [10, 15],
    'min_samples_split': [3, 5],
    'min_samples_leaf': [2],
}

print("[OK] Running GridSearchCV (this may take a moment)...")
grid_search = GridSearchCV(
    base_model, param_grid_small, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1
)
grid_search.fit(X_train_scaled_balanced, y_train_balanced)

print(f"\n[OK] Best parameters found:")
print(f"    {grid_search.best_params_}")
print(f"[OK] Best CV F1 Score: {grid_search.best_score_:.4f}")

# Train final model with best parameters
model = grid_search.best_estimator_

# ============================================================================
# 4. PREDICTIONS
# ============================================================================
print("\n[4] GENERATING PREDICTIONS...")

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)

print(f"[OK] Training accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"[OK] Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

# ============================================================================
# 5. EVALUATION METRICS
# ============================================================================
print("\n[5] EVALUATION METRICS...")

cm = confusion_matrix(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred)
macro_f1 = f1_score(y_test, y_test_pred, average='macro')
weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')
kappa = cohen_kappa_score(y_test, y_test_pred)
mcc = matthews_corrcoef(y_test, y_test_pred)

print(f"\nOVERALL METRICS:")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Macro F1-Score: {macro_f1:.4f}")
print(f"  Weighted F1-Score: {weighted_f1:.4f}")
print(f"  Cohen's Kappa: {kappa:.4f}")
print(f"  Matthews Correlation Coefficient: {mcc:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# Chi-square test
chi2, p_value, dof, _ = chi2_contingency(cm)
print(f"\nStatistical Significance (Chi-Square):")
print(f"  χ² = {chi2:.4f}, p-value = {p_value:.6f}")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_macro')
print(f"\nCross-Validation (5-fold):")
print(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 6. COMPARISON BEFORE/AFTER
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON: BEFORE vs AFTER OPTIMIZATION")
print("=" * 80)

comparison = {
    'Metric': ['Accuracy', 'Macro F1', 'Cohen\'s Kappa', 'MCC', 'Data Size', 'CV Score (mean)'],
    'Before (1014)': ['80.30%', '81.30%', '0.6999', '0.7001', '1,014', '78.54%'],
    'After (3514+SMOTE)': [f'{accuracy*100:.2f}%', f'{macro_f1:.4f}', f'{kappa:.4f}', f'{mcc:.4f}', '3,514+SMOTE', f'{cv_scores.mean():.2%}'],
}

comp_df = pd.DataFrame(comparison)
print("\n" + comp_df.to_string(index=False))

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n[7] SAVING RESULTS...")

# Save comparison
with open('OPTIMIZATION_REPORT.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("OPTIMIZED CLASSIFICATION MODEL - RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write("DATA IMPROVEMENTS:\n")
    f.write(f"  Real data: 1,014 patients\n")
    f.write(f"  Synthetic data: 2,500 patients\n")
    f.write(f"  Combined: 3,514 patients\n")
    f.write(f"  With SMOTE balancing: {X_train_scaled_balanced.shape[0]} samples for training\n\n")
    
    f.write("OPTIMIZATION TECHNIQUES:\n")
    f.write(f"  1. Combined real + synthetic data\n")
    f.write(f"  2. SMOTE for class imbalance\n")
    f.write(f"  3. Hyperparameter tuning with GridSearchCV\n")
    f.write(f"  4. Best parameters: {grid_search.best_params_}\n\n")
    
    f.write("RESULTS:\n")
    f.write(comp_df.to_string())
    f.write(f"\n\nPer-Class Performance:\n")
    f.write(classification_report(y_test, y_test_pred, target_names=le.classes_))

print("[OK] Saved: OPTIMIZATION_REPORT.txt")

print("\n✅ OPTIMIZATION COMPLETE")
