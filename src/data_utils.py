"""Dataset loading, preprocessing, and splitting utilities."""

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DatasetType = Literal['breast_cancer', 'titanic', 'heart_disease']


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    dataset_type: DatasetType = 'breast_cancer'
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
        if self.config.dataset_type == 'breast_cancer':
            X, y = self._load_breast_cancer()
        elif self.config.dataset_type == 'titanic':
            X, y = self._load_titanic()
        elif self.config.dataset_type == 'heart_disease':
            X, y = self._load_heart_disease()
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
    
    def _load_breast_cancer(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load breast cancer dataset (binary classification, 30 features)."""
        data = load_breast_cancer()
        return data.data, data.target
    
    def _load_titanic(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the Titanic survival dataset from OpenML."""
        dataset = fetch_openml("titanic", version=1, as_frame=True)
        df = dataset.frame.copy()
        columns = [
            "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"
        ]
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Titanic dataset is missing columns: {missing}")
        df = df[columns]
        df.replace("?", np.nan, inplace=True)
        numeric_cols = ["pclass", "age", "sibsp", "parch", "fare"]
        categorical_cols = ["sex", "embarked"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        for col in categorical_cols:
            mode_value = df[col].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "missing"
            df[col] = df[col].fillna(fill_value).astype(str)
        X = pd.get_dummies(df.drop(columns=["survived"]), columns=categorical_cols, drop_first=True)
        y_raw = df["survived"].astype(str).str.lower()
        mapping = {
            '1': 1, '0': 0,
            'yes': 1, 'no': 0,
            'true': 1, 'false': 0,
            'alive': 1, 'dead': 0
        }
        y = y_raw.map(mapping)
        if y.isna().any():
            raise ValueError("Unable to map some Titanic targets to binary values.")
        return X.to_numpy(dtype=float), y.to_numpy(dtype=int)

    def _load_heart_disease(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess the Statlog heart disease dataset (OpenML id=53)."""
        dataset = fetch_openml(data_id=53, as_frame=True)
        feature_df = dataset.data.copy()
        target = dataset.target.astype(str).str.lower()
        feature_df.replace("?", np.nan, inplace=True)
        feature_df = feature_df.apply(pd.to_numeric, errors='coerce')
        feature_df = feature_df.fillna(feature_df.median())
        mapping = {
            '1': 1, '0': 0,
            'present': 1, 'absent': 0,
            'yes': 1, 'no': 0
        }
        y = target.map(mapping)
        if y.isna().any():
            raise ValueError("Unable to map some heart-disease targets to binary values.")
        return feature_df.to_numpy(dtype=float), y.to_numpy(dtype=int)
