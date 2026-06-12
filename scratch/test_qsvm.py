import os
import sys
import numpy as np
import pandas as pd

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qs_fraud_qsvm.qsvm import QuantumSVC
from qs_fraud_qsvm.data_processing import preprocess_and_split_data

def test_qsvm_dummy():
    print("Testing QuantumSVC class with mock data...")
    n_qubits = 2
    n_samples_train = 6
    n_samples_test = 4
    
    # Generate mock features (2D, values in range [-1, 1])
    np.random.seed(42)
    X_train = np.random.uniform(-1, 1, (n_samples_train, n_qubits))
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    X_test = np.random.uniform(-1, 1, (n_samples_test, n_qubits))
    y_test = np.array([0, 0, 1, 1])
    
    # Initialize QuantumSVC with 2 qubits and angle embedding
    qsvc = QuantumSVC(n_qubits=n_qubits, embedding_type="angle", C=1.0)
    
    # Fit the model
    print("Fitting model...")
    qsvc.fit(X_train, y_train)
    
    # Test predictions
    print("Making predictions...")
    preds = qsvc.predict(X_test)
    probs = qsvc.predict_proba(X_test)
    decision = qsvc.decision_function(X_test)
    
    print(f"Predictions: {preds}")
    print(f"Probabilities:\n{probs}")
    print(f"Decision function: {decision}")
    
    assert len(preds) == n_samples_test, "Prediction shape mismatch"
    assert probs.shape == (n_samples_test, 2), "Probability shape mismatch"
    assert len(decision) == n_samples_test, "Decision function shape mismatch"
    print("QuantumSVC mock test passed successfully!")

def test_preprocessing_dummy():
    print("\nTesting preprocessing with dummy DataFrame...")
    # Create dummy fraud dataframe
    np.random.seed(42)
    data = {
        'Time': np.random.uniform(0, 100000, 100),
        'Amount': np.random.uniform(0, 500, 100),
        'Class': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    }
    # Add V1 to V28
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, 100)
        
    df = pd.DataFrame(data)
    
    X_train, X_test, y_train, y_test = preprocess_and_split_data(
        df, test_size=0.3, random_state=42,
        n_train_samples=20, n_test_samples=10,
        fraud_ratio=0.5, n_components=3
    )
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train: {y_train}")
    print(f"y_test: {y_test}")
    
    assert X_train.shape == (20, 3), "X_train shape mismatch"
    assert X_test.shape == (10, 3), "X_test shape mismatch"
    assert len(y_train) == 20, "y_train shape mismatch"
    assert len(y_test) == 10, "y_test shape mismatch"
    print("Preprocessing dummy test passed successfully!")

if __name__ == "__main__":
    test_qsvm_dummy()
    test_preprocessing_dummy()
    print("\nAll unit tests passed successfully!")
