import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import os
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pennylane as qml
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==============================================================================
MODEL_NAME = "QuantumCNN"
MONTHS_TO_GENERATE = [1, 3, 5, 6, 7, 8, 9, 11]
# ==============================================================================

# Quantum Circuit Parameters
n_qubits = 4  # Number of qubits
q_depth = 6  # Depth of quantum circuit
q_delta = 0.01  # Initial spread of random quantum weights

# Initialize quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# --- PATH MODIFICATIONS FOR PVC ---
# PVC_MOUNT_ROOT = Path("/home/lucaslopes/code/fire/wildfire-deep-learning/bkp")
TRAIN_FILE_PATH = "data" / "iberfire_train.csv"
TEST_FILE_PATH = "data" / "iberfire_test.csv"
FULL_2024_PATH = "data" / "iberfire_2024.csv"

FIRERISK_MAPS_DIR = "fire_risk_map" / f"{MODEL_NAME}"
FIGURES_DIR = "images" / f"{MODEL_NAME}"
METRICS_DIR = "metrics"
CHECKPOINTS_DIR = "checkpoints" / f"{MODEL_NAME}"

# Create output directories
FIRERISK_MAPS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"IberFire Risk Mapping Pipeline (using {MODEL_NAME})")
print("=" * 50)

# ============================================================================
# QUANTUM CIRCUIT DEFINITION
# ============================================================================


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates."""
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT."""
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_net(q_input_features, q_weights_flat):
    """The variational quantum circuit."""
    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+>, unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    return [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]


class DressedQuantumNet(nn.Module):
    """Dressed quantum circuit with classical pre/post-processing."""

    def __init__(self, input_size=64):
        super().__init__()
        self.pre_net = nn.Linear(input_size, n_qubits)
        self.q_params = nn.Parameter(
            q_delta * torch.randn(q_depth * n_qubits, dtype=torch.float32)
        )
        self.post_net = nn.Linear(n_qubits, 1)

    def forward(self, input_features):
        """Forward pass through the dressed quantum net."""
        # Ensure input is float32
        input_features = input_features.float()

        # Classical pre-processing
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply quantum circuit to each element in the batch
        q_out_list = []
        for elem in q_in:
            # Ensure quantum circuit inputs and weights are float32
            q_out_elem = torch.stack(quantum_net(elem.float(), self.q_params.float()))
            # Explicitly convert output to float32
            q_out_elem = q_out_elem.to(dtype=torch.float32)
            q_out_list.append(q_out_elem)

        q_out = torch.stack(q_out_list)

        # Classical post-processing
        return self.post_net(q_out)


# ============================================================================
# HYBRID QUANTUM-CLASSICAL CNN MODEL
# ============================================================================


class QuantumHybridCNN(nn.Module):
    """CNN with quantum circuit replacing final classification layers."""

    def __init__(self, n_features):
        super(QuantumHybridCNN, self).__init__()

        # Classical CNN feature extraction (frozen after pre-training)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )

        self.flatten = nn.Flatten()

        # Calculate flattened size
        dummy_input = torch.randn(1, 1, n_features)
        x = self.conv_block1(dummy_input)
        x = self.conv_block2(x)
        flattened_size = x.shape[1] * x.shape[2]

        # Classical intermediate layer to reduce dimensions
        self.classical_fc = nn.Sequential(nn.Linear(flattened_size, 64), nn.ReLU())

        # Dressed quantum circuit for final classification
        self.quantum_layer = DressedQuantumNet(input_size=64)

        # Final activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Reshape input
        x = x.unsqueeze(1)

        # Classical CNN feature extraction
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.classical_fc(x)

        # Quantum layer
        x = self.quantum_layer(x)
        x = self.sigmoid(x)

        return x


# ============================================================================
# PYTORCH WRAPPER WITH QUANTUM TRANSFER LEARNING
# ============================================================================


class QuantumCNNWrapper:
    """Wrapper for quantum hybrid CNN with transfer learning."""

    def __init__(self, model_class, n_features, epochs=100, batch_size=512, lr=0.001):
        self.model_class = model_class
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Quantum CNN is using device: {self.device}")
        self.scaler = StandardScaler()
        self.model = None
        self.best_model_state = None
        self.best_loss = float("inf")

    def fit(self, X, y):
        # Initialize model
        self.model = self.model_class(self.n_features).to(self.device)
        self.best_model_state = None
        self.best_loss = float("inf")

        # Freeze classical layers for transfer learning (optional)
        # Uncomment to freeze CNN layers and only train quantum layer
        # for param in self.model.conv_block1.parameters():
        #     param.requires_grad = False
        # for param in self.model.conv_block2.parameters():
        #     param.requires_grad = False

        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"    Training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            total_samples = 0

            # Create progress bar for batches
            pbar = tqdm(
                loader,
                desc=f"    Epoch {epoch + 1}/{self.epochs}",
                leave=False,
                ncols=100,
            )

            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                # Update progress bar with current loss
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_loss = running_loss / total_samples

            # Save best model
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                best_marker = " ✓ (best)"
            else:
                best_marker = ""

            # Print epoch summary
            print(
                f"    Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.4f}{best_marker}"
            )

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n    Loaded best model state with loss: {self.best_loss:.4f}")

        return self

    def predict_proba(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probabilities = self.model(X_tensor).cpu().numpy().flatten()
        return np.vstack([1 - probabilities, probabilities]).T

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities > 0.5).astype(int)

    def save_checkpoint(self, path):
        """Save model checkpoint."""
        if self.best_model_state is not None:
            checkpoint = {
                "model_state_dict": self.best_model_state,
                "scaler_mean": self.scaler.mean_,
                "scaler_scale": self.scaler.scale_,
                "n_features": self.n_features,
                "best_loss": self.best_loss,
                "model_name": MODEL_NAME,
            }
            torch.save(checkpoint, path)
            print(f"\nSuccessfully saved model checkpoint to {path}")
        else:
            print("\nWarning: No best model state found to save.")


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

print("Loading and preparing datasets...")
train_data = pd.read_csv(TRAIN_FILE_PATH)
test_data = pd.read_csv(TEST_FILE_PATH)

features_to_drop = [
    "time",
    "is_fire",
    "is_near_fire",
    "x_index",
    "y_index",
    "x_coordinate",
    "y_coordinate",
    "dim_0",
    "x",
    "y",
]
target_column = "is_fire"

X_train = train_data.drop(columns=features_to_drop, errors="ignore")
y_train = train_data[target_column]
X_test = test_data.drop(columns=features_to_drop, errors="ignore")
y_test = test_data[target_column]

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
n_features = X_train.shape[1]

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

# Train model (define epochs here)
print(f"\nTraining {MODEL_NAME} model on training data...")
final_model = QuantumCNNWrapper(QuantumHybridCNN, n_features=n_features, epochs=30)
final_model.fit(X_train, y_train)

# Save checkpoint
checkpoint_path = CHECKPOINTS_DIR / "best_model.pt"
final_model.save_checkpoint(checkpoint_path)

# Evaluate on test set
print(f"\nEvaluating {MODEL_NAME} on the 2024 test set...")
test_proba = final_model.predict_proba(X_test)[:, 1]
test_preds = (test_proba > 0.5).astype(int)

test_auc = roc_auc_score(y_test, test_proba)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)
print(f"  Test AUC: {test_auc:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  Test F1-Score: {test_f1:.4f}")

# Save metrics
print(f"\nSaving performance metrics to file...")
metrics_file_path = METRICS_DIR / f"{MODEL_NAME}.txt"
with open(metrics_file_path, "w") as f:
    f.write(f"Performance Metrics for Model: {MODEL_NAME}\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 50 + "\n\n")

    f.write("Evaluation on 2024 Test Set\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Test AUC:      {test_auc:.4f}\n")
    f.write(f"  Test Accuracy: {test_acc:.4f}\n")
    f.write(f"  Test F1-Score: {test_f1:.4f}\n")
print(f"    Metrics successfully saved to {metrics_file_path}")

print("\n" + "=" * 50)
print("Pipeline completed!")
