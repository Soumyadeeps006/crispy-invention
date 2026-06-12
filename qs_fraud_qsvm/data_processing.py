import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_cache_dataset(data_dir="data", cache_filename="creditcard_cached.csv.gz"):
    """
    Fetches the Credit Card Fraud dataset from OpenML (ID: 1597) and caches it locally
    as a compressed CSV file. If the cache exists, loads it from there.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, cache_filename)

    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        df = pd.read_csv(cache_path)
    else:
        print("Fetching dataset from OpenML (ID: 1597)... This may take a few minutes...")
        # Fetch using openml ID 1597
        dataset = fetch_openml(data_id=1597, as_frame=True, parser='auto')
        df = dataset.frame
        
        # Ensure target is numeric (OpenML might return it as a category)
        df['Class'] = df['Class'].astype(int)
        
        print(f"Caching dataset to {cache_path}...")
        df.to_csv(cache_path, index=False, compression="gzip")
        print("Caching complete.")
        
    return df

def preprocess_and_split_data(df, test_size=0.2, random_state=42, 
                               n_train_samples=200, n_test_samples=100,
                               fraud_ratio=0.5, n_components=4):
    """
    Preprocesses the dataset:
    1. Splits the data into Train and Test sets first to prevent data leakage.
    2. Performs downsampling on Train and Test sets independently to create balanced subsets.
    3. Scales the features using StandardScaler (fit on train, transform test).
    4. Applies PCA to reduce feature dimensionality to n_components (fit on train, transform test).
    
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed and reduced feature/target matrices.
    """
    # 1. Check for missing values
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        print(f"Warning: Found {null_counts} missing values. Dropping rows with missing values.")
        df = df.dropna()
        
    # Split into features and target
    X = df.drop(columns=['Class'])
    y = df['Class']

    # 2. Strict Train-Test Split first
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Full Train shape: {X_train_full.shape}, Full Test shape: {X_test_full.shape}")
    print(f"Full Train Fraud: {sum(y_train_full == 1)}, Full Test Fraud: {sum(y_test_full == 1)}")

    # Helper function to downsample a dataset maintaining fraud ratio
    def downsample_subset(X_full, y_full, n_total, ratio):
        n_fraud_requested = int(n_total * ratio)
        n_legit_requested = n_total - n_fraud_requested

        # Separate indices
        fraud_idx = y_full[y_full == 1].index
        legit_idx = y_full[y_full == 0].index

        # Sample fraud
        n_fraud = min(n_fraud_requested, len(fraud_idx))
        np.random.seed(random_state)
        fraud_sampled_idx = np.random.choice(fraud_idx, size=n_fraud, replace=False)

        # Sample legit
        n_legit = min(n_legit_requested + (n_fraud_requested - n_fraud), len(legit_idx))
        legit_sampled_idx = np.random.choice(legit_idx, size=n_legit, replace=False)

        # Combine and shuffle
        combined_idx = np.concatenate([fraud_sampled_idx, legit_sampled_idx])
        np.random.shuffle(combined_idx)

        return X_full.loc[combined_idx], y_full.loc[combined_idx]

    # 3. Downsample Train and Test independently to prevent leakage and match computation bounds
    X_train_down, y_train_down = downsample_subset(X_train_full, y_train_full, n_train_samples, fraud_ratio)
    X_test_down, y_test_down = downsample_subset(X_test_full, y_test_full, n_test_samples, fraud_ratio)

    print(f"Downsampled Train shape: {X_train_down.shape} (Fraud: {sum(y_train_down == 1)})")
    print(f"Downsampled Test shape: {X_test_down.shape} (Fraud: {sum(y_test_down == 1)})")

    # 4. Fit Preprocessing (Scaler and PCA) on Train, and Transform Test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_down)
    X_test_scaled = scaler.transform(X_test_down)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Convert targets to numpy arrays for compatibility
    y_train_arr = y_train_down.values
    y_test_arr = y_test_down.values

    # Check explained variance
    explained_var = sum(pca.explained_variance_ratio_)
    print(f"PCA with {n_components} components explains {explained_var:.2%} of variance.")

    return X_train_pca, X_test_pca, y_train_arr, y_test_arr
