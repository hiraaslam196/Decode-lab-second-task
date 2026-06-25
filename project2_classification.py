"""
Project 2: Data Classification Using AI
Batch: 2026 | DecodeLabs

A complete implementation of a supervised learning classification model
using the Iris dataset and K-Nearest Neighbors algorithm.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: LOAD AND UNDERSTAND THE DATASET
# ============================================
print("="*60)
print("PROJECT 2: DATA CLASSIFICATION USING AI")
print("Powered by DecodeLabs | Batch: 2026")
print("="*60)

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels
feature_names = iris.feature_names
target_names = iris.target_names

print("\n📊 DATASET INFORMATION")
print("-"*40)
print(f"Total Samples: {len(X)} (Balanced)")
print(f"Number of Classes: {len(target_names)}")
print(f"Features/Dimensions: {X.shape[1]}")
print(f"Classes: {target_names}")
print(f"Feature Names: {feature_names}")

# Display first 5 rows
print("\n📋 First 5 Samples:")
df = pd.DataFrame(X, columns=feature_names)
df['Species'] = y
print(df.head())

# Check class distribution
print("\n📈 Class Distribution:")
for i, name in enumerate(target_names):
    count = np.sum(y == i)
    print(f"  {name}: {count} samples")

# ============================================
# STEP 2: DATA PREPROCESSING - SCALING
# ============================================
print("\n⚙️ DATA PREPROCESSING")
print("-"*40)

# StandardScaler: Mean = 0, Variance = 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Feature scaling applied using StandardScaler")
print(f"   Mean of scaled features: {X_scaled.mean(axis=0).round(6)}")
print(f"   Variance of scaled features: {X_scaled.var(axis=0).round(6)}")

# ============================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================
print("\n🔀 TRAIN-TEST SPLIT")
print("-"*40)

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print(f"Train-Test Ratio: {len(X_train)/(len(X_train)+len(X_test)):.1%} / {len(X_test)/(len(X_train)+len(X_test)):.1%}")

# ============================================
# STEP 4: K-NEAREST NEIGHBORS ALGORITHM
# ============================================
print("\n🧠 K-NEAREST NEIGHBORS MODEL")
print("-"*40)

# Find optimal K value using cross-validation
k_range = range(1, 31)
k_scores = []
k_f1_scores = []

from sklearn.model_selection import cross_val_score

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Cross-validation accuracy
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
    # Cross-validation F1
    f1_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1_macro')
    k_f1_scores.append(f1_scores.mean())

# Find best K
best_k = k_range[np.argmax(k_scores)]
best_f1 = max(k_f1_scores)

print(f"📊 Optimal K value: {best_k}")
print(f"   Best Cross-Validation Accuracy: {max(k_scores):.4f}")
print(f"   Best Cross-Validation F1 Score: {best_f1:.4f}")

# Plot K selection
plt.figure(figsize=(10, 5))
plt.plot(k_range, k_scores, 'bo-', label='Accuracy')
plt.plot(k_range, k_f1_scores, 'ro-', label='F1 Score')
plt.xlabel('K Value')
plt.ylabel('Cross-Validation Score')
plt.title('K Value Selection for KNN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('k_selection.png', dpi=100, bbox_inches='tight')
print("✅ K-selection plot saved as 'k_selection.png'")

# ============================================
# STEP 5: TRAIN THE MODEL
# ============================================
print("\n🎯 MODEL TRAINING")
print("-"*40)

# Instantiate and train the model with best K
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")
print(f"   Algorithm: KNN (K={best_k})")
print(f"   Training completed on {len(X_train)} samples")

# ============================================
# STEP 6: MAKE PREDICTIONS
# ============================================
print("\n🔮 MAKING PREDICTIONS")
print("-"*40)

# Predict on test set
y_pred = model.predict(X_test)

print("✅ Predictions completed on test set")

# ============================================
# STEP 7: EVALUATE MODEL PERFORMANCE
# ============================================
print("\n📊 MODEL EVALUATION")
print("-"*40)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Test Set Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1 Score (Macro): {f1:.4f}")

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# ============================================
# STEP 8: CONFUSION MATRIX
# ============================================
print("\n📊 CONFUSION MATRIX")
print("-"*40)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title('Confusion Matrix - Iris Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
print("✅ Confusion matrix saved as 'confusion_matrix.png'")

# ============================================
# STEP 9: VISUALIZE DECISION BOUNDARIES
# ============================================
print("\n🎨 VISUALIZING DECISION BOUNDARIES")
print("-"*40)

# Create 2D visualization using first two features
plt.figure(figsize=(12, 5))

# Plot 1: First 2 features
plt.subplot(1, 2, 1)
for i, name in enumerate(target_names):
    mask = y == i
    plt.scatter(X[mask, 0], X[mask, 1], label=name, alpha=0.7)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Iris Dataset - First 2 Features')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Last 2 features
plt.subplot(1, 2, 2)
for i, name in enumerate(target_names):
    mask = y == i
    plt.scatter(X[mask, 2], X[mask, 3], label=name, alpha=0.7)
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.title('Iris Dataset - Last 2 Features')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_visualization.png', dpi=100, bbox_inches='tight')
print("✅ Feature visualization saved as 'feature_visualization.png'")

# ============================================
# STEP 10: SAMPLE PREDICTION
# ============================================
print("\n🔍 SAMPLE PREDICTIONS")
print("-"*40)

# Show some sample predictions with actual values
print("Sample predictions vs actual:")
print("-"*30)
sample_indices = np.random.choice(len(X_test), 5, replace=False)
for i, idx in enumerate(sample_indices):
    actual = target_names[y_test[idx]]
    predicted = target_names[y_pred[idx]]
    status = "✅" if actual == predicted else "❌"
    print(f"Sample {i+1}: Actual={actual}, Predicted={predicted} {status}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("📌 PROJECT SUMMARY")
print("="*60)
print(f"""
Dataset: Iris (150 samples, 3 classes, 4 features)
Model: K-Nearest Neighbors (K={best_k})
Training Size: {len(X_train)} samples
Testing Size: {len(X_test)} samples
Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
F1 Score: {f1:.4f}

✅ Project completed successfully!
📁 Generated files:
   - k_selection.png
   - confusion_matrix.png  
   - feature_visualization.png
""")

print("="*60)
print("Thank you for completing Project 2!")
print("DecodeLabs | Batch 2026")
print("="*60)
