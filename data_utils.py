"""
Data Management Utilities

This module handles dataset generation, preprocessing, and configuration.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal
from sklearn.datasets import (
    make_classification, 
    load_breast_cancer, 
    load_iris,
    load_wine,
    load_digits
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DatasetType = Literal['synthetic', 'breast_cancer', 'iris', 'wine', 'digits']


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    dataset_type: DatasetType = 'synthetic'
    # Synthetic dataset parameters
    n_samples: int = 1000
    n_features: int = 20
    n_informative: int = 6
    n_redundant: int = 8
    n_repeated: int = 4
    n_clusters_per_class: int = 2
    class_sep: float = 0.5
    flip_y: float = 0.15
    # Common parameters
    test_size: float = 0.2
    random_state: int = 42


class DataManager:
    """
    Handles data generation, preprocessing, and splitting.
    Follows Single Responsibility Principle.
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.scaler = StandardScaler()
    
    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load or generate and preprocess classification dataset.
        Returns: X_train, X_test, y_train, y_test
        """
        # Load dataset based on type
        if self.config.dataset_type == 'synthetic':
            X, y = self._load_synthetic()
        elif self.config.dataset_type == 'breast_cancer':
            X, y = self._load_breast_cancer()
        elif self.config.dataset_type == 'iris':
            X, y = self._load_iris()
        elif self.config.dataset_type == 'wine':
            X, y = self._load_wine()
        elif self.config.dataset_type == 'digits':
            X, y = self._load_digits()
        else:
            raise ValueError(f"Unknown dataset type: {self.config.dataset_type}")
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        print(f"Dataset: {self.config.dataset_type}")
        print(f"Shape: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"Classes: {np.unique(y)}")
        print("-" * 60)
        
        return X_train, X_test, y_train, y_test
    
    def _load_synthetic(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification dataset."""
        X, y = make_classification(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            n_informative=self.config.n_informative,
            n_redundant=self.config.n_redundant,
            n_repeated=self.config.n_repeated,
            n_clusters_per_class=self.config.n_clusters_per_class,
            class_sep=self.config.class_sep,
            flip_y=self.config.flip_y,
            random_state=self.config.random_state
        )
        return X, y
    
    def _load_breast_cancer(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load breast cancer dataset (binary classification, 30 features)."""
        data = load_breast_cancer()
        return data.data, data.target
    
    def _load_iris(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load iris dataset (3 classes, 4 features). Convert to binary."""
        data = load_iris()
        # Convert to binary: class 0 vs rest
        y_binary = (data.target == 0).astype(int)
        return data.data, y_binary
    
    def _load_wine(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load wine dataset (3 classes, 13 features). Convert to binary."""
        data = load_wine()
        # Convert to binary: class 0 vs rest
        y_binary = (data.target == 0).astype(int)
        return data.data, y_binary
    
    def _load_digits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load digits dataset (10 classes, 64 features). Convert to binary."""
        data = load_digits()
        # Convert to binary: digits 0-4 vs 5-9
        y_binary = (data.target < 5).astype(int)
        return data.data, y_binary
