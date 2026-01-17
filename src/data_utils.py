"""Dataset loading, preprocessing, and splitting utilities."""

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DatasetType = Literal[
    'breast_cancer', 'titanic', 'heart_disease', 'banknote',
    'pima_diabetes', 'german_credit', 'adult_income', 'higgs_small'
]


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
        elif self.config.dataset_type == 'banknote':
            X, y = self._load_banknote()
        elif self.config.dataset_type == 'pima_diabetes':
            X, y = self._load_pima_diabetes()
        elif self.config.dataset_type == 'german_credit':
            X, y = self._load_german_credit()
        elif self.config.dataset_type == 'adult_income':
            X, y = self._load_adult_income()
        elif self.config.dataset_type == 'higgs_small':
            X, y = self._load_higgs_small()
        else:
            raise ValueError(f"Unknown dataset type: {self.config.dataset_type}")
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split into train and test (stratified to ensure both classes in each split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y
        )
        
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

    def _load_banknote(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the Banknote Authentication dataset (OpenML id=1462).
        
        Binary classification with 4 features extracted from wavelet-transformed
        images of banknotes: variance, skewness, curtosis, and entropy.
        Target: 0 = authentic (class 1 in raw data), 1 = forged (class 2 in raw data).
        """
        dataset = fetch_openml(data_id=1462, as_frame=True)
        feature_df = dataset.data.copy()
        target = dataset.target.astype(str).str.strip()
        feature_df = feature_df.apply(pd.to_numeric, errors='coerce')
        feature_df = feature_df.fillna(feature_df.median())
        mapping = {'1': 0, '2': 1}  # Class 1 = authentic (0), Class 2 = forged (1)
        y = target.map(mapping)
        if y.isna().any():
            raise ValueError("Unable to map some banknote targets to binary values.")
        return feature_df.to_numpy(dtype=float), y.to_numpy(dtype=int)

    def _load_pima_diabetes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the Pima Indians Diabetes dataset (OpenML id=37).
        
        Binary classification with 8 features related to health indicators.
        Target: 0 = no diabetes, 1 = diabetes.
        """
        dataset = fetch_openml(data_id=37, as_frame=True)
        feature_df = dataset.data.copy()
        target = dataset.target.astype(str).str.strip()
        feature_df = feature_df.apply(pd.to_numeric, errors='coerce')
        feature_df = feature_df.fillna(feature_df.median())
        mapping = {'tested_negative': 0, 'tested_positive': 1, '0': 0, '1': 1}
        y = target.map(mapping)
        if y.isna().any():
            raise ValueError("Unable to map some pima diabetes targets to binary values.")
        return feature_df.to_numpy(dtype=float), y.to_numpy(dtype=int)

    def _load_german_credit(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the German Credit Risk dataset (OpenML id=31).
        
        Binary classification with 20 features about credit applicants.
        Target: 0 = bad credit, 1 = good credit.
        """
        dataset = fetch_openml(data_id=31, as_frame=True)
        feature_df = dataset.data.copy()
        target = dataset.target.astype(str).str.strip()
        
        # Identify numeric and categorical columns
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Handle numeric columns
        for col in numeric_cols:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        
        # One-hot encode categorical columns
        if categorical_cols:
            feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=True)
        
        mapping = {'good': 1, 'bad': 0, '1': 1, '2': 0}
        y = target.map(mapping)
        if y.isna().any():
            raise ValueError("Unable to map some german credit targets to binary values.")
        return feature_df.to_numpy(dtype=float), y.to_numpy(dtype=int)

    def _load_adult_income(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the Adult (Census Income) dataset (OpenML id=1590).
        
        Binary classification predicting if income exceeds $50K/year.
        Target: 0 = <=50K, 1 = >50K.
        """
        dataset = fetch_openml(data_id=1590, as_frame=True)
        feature_df = dataset.data.copy()
        target = dataset.target.astype(str).str.strip()
        
        # Identify numeric and categorical columns
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Handle numeric columns
        for col in numeric_cols:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        
        # Handle categorical columns - fill missing and one-hot encode
        for col in categorical_cols:
            mode_value = feature_df[col].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "missing"
            feature_df[col] = feature_df[col].fillna(fill_value).astype(str)
        
        if categorical_cols:
            feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=True)
        
        mapping = {'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1, '0': 0, '1': 1}
        y = target.map(mapping)
        if y.isna().any():
            raise ValueError("Unable to map some adult income targets to binary values.")
        return feature_df.to_numpy(dtype=float), y.to_numpy(dtype=int)

    def _load_higgs_small(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load a small subset of the Higgs Boson dataset (OpenML id=23512).
        
        Binary classification for detecting Higgs bosons from background noise.
        Uses first 10,000 samples to keep computation manageable.
        Target: 0 = background, 1 = signal (Higgs boson).
        """
        dataset = fetch_openml(data_id=23512, as_frame=True)
        feature_df = dataset.data.copy()
        target = dataset.target.astype(str).str.strip()
        
        # Take a small subset (10,000 samples) for faster training
        max_samples = 10000
        if len(feature_df) > max_samples:
            # Use a deterministic subset based on index
            feature_df = feature_df.iloc[:max_samples]
            target = target.iloc[:max_samples]
        
        feature_df = feature_df.apply(pd.to_numeric, errors='coerce')
        feature_df = feature_df.fillna(feature_df.median())
        
        mapping = {'0': 0, '1': 1, '0.0': 0, '1.0': 1}
        y = target.map(mapping)
        if y.isna().any():
            raise ValueError("Unable to map some higgs targets to binary values.")
        return feature_df.to_numpy(dtype=float), y.to_numpy(dtype=int)
