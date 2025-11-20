# Imports, as always...
import pandas as pd

import torch
from torch.utils.data import Dataset

from typing import Literal

# Options for the feature vector.
FeatureType = Literal[
    'raw', 'res', 'abs_res', 'rel_res', 'abs_rel_res', 'log_ratio', 'qubit_res', 'abs_qubit_res', 'pauli_z_res', 'pauli_zz_res']


class QuantumDataset(Dataset):
    """Custom dataset for handling our measurement data and computing features."""

    def __init__(self, df: pd.DataFrame, feature_type: FeatureType = 'raw'):
        self.df = df.copy()
        self.feature_type = feature_type

        # Assuming fixed n.
        self.n = self.df['n'].iloc[0]

        # Encoding device labels.
        device_map = {device: i for i, device in enumerate(self.df['device'].unique())}
        self.df['label'] = self.df['device'].apply(lambda x: device_map[x])
        self.labels = torch.tensor(self.df['label'].values, dtype=torch.long)

        # Precompute features.
        self.features = self._precompute_features()

        # Standardisation.
        mean = self.features.mean(dim=0)
        std = self.features.std(dim=0)
        self.features = (self.features - mean) / (std + 1e-6)

    def _precompute_features(self) -> torch.Tensor:
        # Small value for div-by-zero handling.
        eps = 1e-6

        # Relevant columns.
        x_cols = [col for col in self.df.columns if col.startswith('x')]
        y_cols = [col for col in self.df.columns if col.startswith('y')]

        # Relevant values.
        x_probs = torch.tensor(self.df[x_cols].values, dtype=torch.float64)
        y_probs = torch.tensor(self.df[y_cols].values, dtype=torch.float64)

        # Per-outcome feature types...

        if self.feature_type == 'raw':
            return x_probs

        if self.feature_type == 'res':
            return x_probs - y_probs

        if self.feature_type == 'abs_res':
            return (x_probs - y_probs).abs()

        if self.feature_type == 'rel_res':
            return (x_probs - y_probs) / (y_probs + eps)

        if self.feature_type == 'abs_rel_res':
            return (x_probs - y_probs).abs() / (y_probs + eps)

        if self.feature_type == 'log_ratio':
            return ((x_probs + eps) / (y_probs + eps)).log()

        # Per-qubit feature types...

        if self.feature_type == 'qubit_res':
            n_samples = x_probs.shape[0]
            residuals = x_probs - y_probs
            qubit_residuals = torch.zeros(n_samples, self.n, dtype=x_probs.dtype)

            for i in range(self.n):
                qubit_mask = torch.tensor([(j >> i) & 1 == 0 for j in range(2 ** self.n)], dtype=torch.bool)
                qubit_residuals[:, i] = residuals[:, qubit_mask].sum(dim=1)

            return qubit_residuals

        if self.feature_type == 'abs_qubit_res':
            n_samples = x_probs.shape[0]
            residuals = (x_probs - y_probs).abs()
            qubit_residuals = torch.zeros(n_samples, self.n, dtype=x_probs.dtype)

            for i in range(self.n):
                qubit_mask = torch.tensor([(j >> i) & 1 == 0 for j in range(2 ** self.n)], dtype=torch.bool)
                qubit_residuals[:, i] = residuals[:, qubit_mask].sum(dim=1)

            return qubit_residuals

        if self.feature_type == 'pauli_z_res':
            n_samples = x_probs.shape[0]
            residuals = x_probs - y_probs
            pauli_z_residuals = torch.zeros(n_samples, self.n, dtype=x_probs.dtype)

            for i in range(self.n):
                sign_vector = torch.tensor([1 - 2 * ((j >> i) & 1) for j in range(2 ** self.n)], dtype=x_probs.dtype)
                weighted_residual = residuals * sign_vector
                pauli_z_residuals[:, i] = weighted_residual.sum(dim=1)

            return pauli_z_residuals

        if self.feature_type == 'pauli_zz_res':
            n_samples = x_probs.shape[0]
            residuals = x_probs - y_probs

            n_pairs = self.n * (self.n - 1) // 2
            pauli_zz_residuals = torch.zeros(n_samples, n_pairs, dtype=x_probs.dtype)

            pair_idx = 0

            for i in range(self.n):
                for j in range(i + 1, self.n):
                    sign_vector = torch.zeros(32, dtype=x_probs.dtype)
                    for b in range(2 ** self.n):
                        b_i = (b >> i) & 1
                        b_j = (b >> j) & 1

                        if b_i == b_j:
                            sign_vector[b] = 1.0
                        else:
                            sign_vector[b] = -1.0

                    weighted_residual = residuals * sign_vector
                    pauli_zz_residuals[:, pair_idx] = weighted_residual.sum(dim=1)
                    pair_idx += 1

            return pauli_zz_residuals

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
