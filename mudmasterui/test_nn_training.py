"""
Test Neural Network Training
"""

import numpy as np
from models.nn_trainer import NeuralNetworkTrainer

# Generate synthetic data
print("Generating synthetic training data...")
n_samples = 100
n_features = 1077  # 359 frequencies × 3 (epsilon_real, epsilon_imag, loss_tangent)

# Create features with some pattern
X = np.random.randn(n_samples, n_features)

# Create target with some relationship to features
# Simple linear relationship + noise for demonstration
y = (X[:, :10].mean(axis=1) * 10 + 
     np.random.randn(n_samples) * 2 + 20)

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Target range: {y.min():.2f} - {y.max():.2f}")

# Initialize trainer
trainer = NeuralNetworkTrainer(depth_cm=100)

# Train model
print("\n" + "="*70)
results = trainer.train(
    X, y,
    hidden_layers=(128, 64, 32),
    max_iter=500,
    verbose=True
)

# Save model
print("\n" + "="*70)
paths = trainer.save(name="test_model")

# Test prediction
print("\n" + "="*70)
print("Testing Predictions")
print("="*70)

# Predict on first 5 samples
X_test = X[:5]
y_true = y[:5]
y_pred = trainer.predict(X_test)

print("\nSample predictions:")
for i in range(5):
    print(f"  Sample {i+1}: True={y_true[i]:.2f}, Predicted={y_pred[i]:.2f}, Error={abs(y_true[i]-y_pred[i]):.2f}")

print("\n✓ Training and prediction successful!")
