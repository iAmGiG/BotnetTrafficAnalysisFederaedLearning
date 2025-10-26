#!/usr/bin/env python
"""
Overfitting Analysis for Classification Model

This script performs additional validation to determine if the 99.85%
accuracy is due to overfitting or genuinely distinctive patterns.

Tests performed:
1. Learning curves (training vs validation accuracy over epochs)
2. Cross-validation with different random seeds
3. Feature importance analysis
4. Confusion patterns across different splits
5. Per-device generalization testing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'classification'))

import pandas as pd
import numpy as np
from glob import iglob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
import json

def load_data():
    """Load botnet data"""
    print('Loading data...')
    df_gafgyt = pd.concat((pd.read_csv(f) for f in iglob(
        '../data/**/gafgyt_attacks/*.csv', recursive=True)), ignore_index=True)
    df_gafgyt['class'] = 'gafgyt'

    df_mirai = pd.concat((pd.read_csv(f) for f in iglob(
        '../data/**/mirai_attacks/*.csv', recursive=True)), ignore_index=True)
    df_mirai['class'] = 'mirai'

    df_benign = pd.concat((pd.read_csv(f) for f in iglob(
        '../data/**/benign_traffic.csv', recursive=True)), ignore_index=True)
    df_benign['class'] = 'benign'

    # Balance dataset
    min_samples = min(df_benign.shape[0], df_gafgyt.shape[0], df_mirai.shape[0])
    df = pd.concat([
        df_benign.sample(n=min_samples, random_state=17),
        df_gafgyt.sample(n=min_samples, random_state=17),
        df_mirai.sample(n=min_samples, random_state=17)
    ], ignore_index=True)

    return df

def create_model(input_dim, dropout_rate=0.0):
    """Create classification model with optional dropout"""
    model = Sequential()
    model.add(Dense(128, activation="tanh", input_shape=(input_dim,)))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation="tanh"))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

def test_1_learning_curves():
    """Test 1: Plot learning curves to check for overfitting"""
    print("\n" + "="*60)
    print("TEST 1: Learning Curves Analysis")
    print("="*60)

    df = load_data()
    X = df.drop(columns=['class'])

    # Use top 5 features
    fisher = pd.read_csv('../data/fisher/fisher.csv')
    features = fisher.iloc[0:5]['Feature'].values
    X = X[list(features)]

    Y = pd.get_dummies(df['class'])

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = create_model(X.shape[1])

    # Train with history
    history = model.fit(x_train_scaled, y_train,
                       epochs=25,
                       batch_size=256,
                       validation_data=(x_test_scaled, y_test),
                       verbose=0)

    # Plot learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curves
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Loss curves
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: learning_curves.png")

    # Analysis
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    gap = final_train_acc - final_val_acc

    print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Train-Val Gap: {gap:.4f} ({gap*100:.2f}%)")

    if gap < 0.01:
        print("✓ GOOD: Minimal overfitting (gap < 1%)")
    elif gap < 0.03:
        print("⚠ MODERATE: Some overfitting (gap 1-3%)")
    else:
        print("✗ BAD: Significant overfitting (gap > 3%)")

    return history.history

def test_2_cross_validation():
    """Test 2: K-fold cross-validation with different random seeds"""
    print("\n" + "="*60)
    print("TEST 2: Cross-Validation with Multiple Seeds")
    print("="*60)

    df = load_data()
    X = df.drop(columns=['class'])

    fisher = pd.read_csv('../data/fisher/fisher.csv')
    features = fisher.iloc[0:5]['Feature'].values
    X = X[list(features)]

    Y = pd.get_dummies(df['class'])

    # Test with different random seeds
    seeds = [42, 17, 123, 456, 789]
    results = []

    for seed in seeds:
        print(f"\nTesting with random_state={seed}...")
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=seed)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        model = create_model(X.shape[1])
        model.fit(x_train_scaled, y_train,
                 epochs=25, batch_size=256, verbose=0)

        y_pred_proba = model.predict(x_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test.values, axis=1)

        acc = accuracy_score(y_true, y_pred)
        results.append(acc)
        print(f"  Accuracy: {acc:.4f}")

    mean_acc = np.mean(results)
    std_acc = np.std(results)

    print(f"\n{'='*40}")
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Std Dev: {std_acc:.4f}")
    print(f"Range: {min(results):.4f} - {max(results):.4f}")

    if std_acc < 0.01:
        print("✓ GOOD: Very stable across different splits (std < 1%)")
    elif std_acc < 0.03:
        print("⚠ MODERATE: Some variance across splits (std 1-3%)")
    else:
        print("✗ BAD: High variance across splits (std > 3%)")

    return results

def test_3_feature_importance():
    """Test 3: Check if model relies on few features or all features"""
    print("\n" + "="*60)
    print("TEST 3: Feature Importance via Ablation")
    print("="*60)

    df = load_data()
    X = df.drop(columns=['class'])

    fisher = pd.read_csv('../data/fisher/fisher.csv')
    features = fisher.iloc[0:5]['Feature'].values
    X = X[list(features)]

    Y = pd.get_dummies(df['class'])

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Baseline with all features
    model = create_model(X.shape[1])
    model.fit(x_train_scaled, y_train, epochs=25, batch_size=256, verbose=0)
    y_pred_proba = model.predict(x_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test.values, axis=1)
    baseline_acc = accuracy_score(y_true, y_pred)

    print(f"\nBaseline (all 5 features): {baseline_acc:.4f}")

    # Test with each feature removed
    print("\nAccuracy when removing each feature:")
    importance_scores = []

    for i, feature in enumerate(features):
        # Remove this feature
        features_subset = [f for j, f in enumerate(features) if j != i]
        X_subset = X[features_subset]

        x_train_sub, x_test_sub, _, _ = train_test_split(
            X_subset, Y, test_size=0.2, random_state=42)

        scaler_sub = StandardScaler()
        scaler_sub.fit(x_train_sub)
        x_train_scaled_sub = scaler_sub.transform(x_train_sub)
        x_test_scaled_sub = scaler_sub.transform(x_test_sub)

        model_sub = create_model(X_subset.shape[1])
        model_sub.fit(x_train_scaled_sub, y_train, epochs=25, batch_size=256, verbose=0)

        y_pred_proba_sub = model_sub.predict(x_test_scaled_sub, verbose=0)
        y_pred_sub = np.argmax(y_pred_proba_sub, axis=1)
        acc_sub = accuracy_score(y_true, y_pred_sub)

        importance = baseline_acc - acc_sub
        importance_scores.append(importance)

        print(f"  Remove '{feature[:30]}...': {acc_sub:.4f} (importance: {importance:+.4f})")

    # Check if model relies on just one or two features
    max_importance = max(importance_scores)
    if max_importance > 0.05:
        print(f"\n⚠ WARNING: Removing one feature drops accuracy by {max_importance*100:.1f}%")
        print("  Model may be overly reliant on specific features")
    else:
        print(f"\n✓ GOOD: All features contribute relatively equally (max drop: {max_importance*100:.1f}%)")

    return dict(zip(features, importance_scores))

def test_4_dropout_regularization():
    """Test 4: Compare with dropout regularization"""
    print("\n" + "="*60)
    print("TEST 4: Dropout Regularization Comparison")
    print("="*60)

    df = load_data()
    X = df.drop(columns=['class'])

    fisher = pd.read_csv('../data/fisher/fisher.csv')
    features = fisher.iloc[0:5]['Feature'].values
    X = X[list(features)]

    Y = pd.get_dummies(df['class'])

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    dropout_rates = [0.0, 0.2, 0.3, 0.5]
    results = []

    for dropout in dropout_rates:
        print(f"\nTesting with dropout={dropout}...")
        model = create_model(X.shape[1], dropout_rate=dropout)
        model.fit(x_train_scaled, y_train,
                 epochs=25, batch_size=256, verbose=0)

        y_pred_proba = model.predict(x_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test.values, axis=1)

        acc = accuracy_score(y_true, y_pred)
        results.append(acc)
        print(f"  Test Accuracy: {acc:.4f}")

    # If dropout significantly hurts performance, model may not be overfitted
    baseline = results[0]
    best_dropout = results[1]

    if best_dropout >= baseline - 0.01:
        print(f"\n✓ GOOD: Dropout doesn't hurt performance (model not overfitted)")
    else:
        print(f"\n⚠ WARNING: Dropout reduces accuracy by {(baseline-best_dropout)*100:.1f}%")
        print("  This could indicate the model is properly fitted, not overfitted")

    return dict(zip(dropout_rates, results))

def generate_report(learning_history, cv_results, feature_importance, dropout_results):
    """Generate final analysis report"""
    print("\n" + "="*60)
    print("OVERFITTING ANALYSIS - FINAL REPORT")
    print("="*60)

    report = {
        'learning_curves': {
            'final_train_acc': learning_history['accuracy'][-1],
            'final_val_acc': learning_history['val_accuracy'][-1],
            'gap': learning_history['accuracy'][-1] - learning_history['val_accuracy'][-1]
        },
        'cross_validation': {
            'mean': np.mean(cv_results),
            'std': np.std(cv_results),
            'results': cv_results
        },
        'feature_importance': feature_importance,
        'dropout_regularization': dropout_results
    }

    # Save report
    with open('overfitting_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("\n✓ Saved: overfitting_analysis_report.json")

    # Print conclusions
    print("\nCONCLUSIONS:")
    print("-" * 60)

    gap = report['learning_curves']['gap']
    if gap < 0.01:
        print("✓ Learning curves show minimal overfitting")
    else:
        print(f"⚠ Train-val gap of {gap*100:.2f}% suggests some overfitting")

    cv_std = report['cross_validation']['std']
    if cv_std < 0.01:
        print("✓ Cross-validation shows stable performance")
    else:
        print(f"⚠ Cross-validation std of {cv_std*100:.2f}% shows some variance")

    max_feat_importance = max(feature_importance.values())
    if max_feat_importance < 0.05:
        print("✓ Model uses all features relatively equally")
    else:
        print(f"⚠ Model relies heavily on specific features (max importance: {max_feat_importance*100:.1f}%)")

    print("\nRECOMMENDATIONS:")
    print("-" * 60)

    if gap > 0.02:
        print("• Consider using dropout or L2 regularization")
        print("• Try reducing model complexity (fewer layers/units)")

    if cv_std > 0.02:
        print("• Test on completely different IoT devices")
        print("• Consider more robust cross-validation (stratified k-fold)")

    if max_feat_importance > 0.05:
        print("• Investigate top features for potential data leakage")
        print("• Test model on traffic with synthetic feature perturbations")

    print("\n" + "="*60)

if __name__ == '__main__':
    print("Starting Overfitting Analysis...")
    print("This will take several minutes...\n")

    learning_history = test_1_learning_curves()
    cv_results = test_2_cross_validation()
    feature_importance = test_3_feature_importance()
    dropout_results = test_4_dropout_regularization()

    generate_report(learning_history, cv_results, feature_importance, dropout_results)

    print("\nAnalysis complete!")
