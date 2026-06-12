# Quantum SVM for Financial Fraud Detection

This project implements a **Quantum Support Vector Machine (QSVM)** to detect financial credit card fraud, using **PennyLane** for quantum simulation and **scikit-learn** for model training and evaluation. It compares the QSVM with classical baseline classifiers: **Classical SVM (RBF)** and **Random Forest**.

## Project Architecture

```
qs-fraud-qsvm/
│
├── qs_fraud_qsvm/                 # Python package
│   ├── __init__.py
│   ├── data_processing.py         # Data downloading, caching, downsampling, and scaling
│   ├── qsvm.py                    # Quantum SVC class using PennyLane QNodes
│   ├── visualization.py           # Evaluation curve and decision boundary plotting
│   └── pipeline.py                # End-to-end command-line runner
│
├── notebooks/
│   └── QSVM_Fraud_Detection.ipynb # Interactive storytelling notebook
│
├── results/                       # Generated evaluation plots and summary metrics
│
├── scratch/                       # Test and generation helper scripts
│
├── pyproject.toml                 # Build config and dependencies
├── requirements.txt               # Direct package dependencies
└── README.md                      # Documentation
```

## Features

1. **Robust Caching**: Automatically downloads the Credit Card Fraud dataset from OpenML (ID: 1597) and caches it locally as a compressed CSV (`data/creditcard_cached.csv.gz`), avoiding repeated slow downloads.
2. **Data Leakage Protection**: Follows strict machine learning best practices. The dataset is split into training and test sets *before* any downsampling, scaling, or dimensionality reduction. Scalers and PCA are fit exclusively on training data and applied to test data.
3. **Quantum Kernel Mapping**: Uses a custom quantum kernel function:
   $$K(x_i, x_j) = |\langle \Phi(x_i) | \Phi(x_j) \rangle|^2$$
   simulated via a 4-qubit circuit with `AngleEmbedding` (rotation on the X axis) and its adjoint.
4. **Precomputed Kernel Optimizations**: Leverages symmetry ($K_{i,j} = K_{j,i}$) and unit diagonals ($K_{i,i} = 1.0$) to reduce quantum circuit simulation runs by 50% during training.
5. **Decision Boundary Projection**: Visualizes the high-dimensional quantum decision boundary by projecting it onto the first 2 principal components of the PCA feature space.

## Setup Instructions

1. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # On Windows
   source .venv/bin/activate   # On Unix/macOS
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Unit Tests**:
   ```bash
   python scratch/test_qsvm.py
   ```

## Running the Pipeline

You can run the end-to-end training and evaluation pipeline via command line:

```bash
python -m qs_fraud_qsvm.pipeline --n_train 200 --n_test 100 --n_qubits 4 --embedding angle
```

### Options
- `--n_train`: Number of training samples (default: `200`)
- `--n_test`: Number of test samples (default: `100`)
- `--fraud_ratio`: Proportion of fraud cases in the sampled subsets (default: `0.5`)
- `--n_qubits`: Number of PCA features / qubits (default: `4`)
- `--C`: SVM regularization parameter (default: `1.0`)
- `--embedding`: Quantum feature map embedding type, either `angle` or `iqp` (default: `angle`)

All output metrics and plots are saved in the `results/` folder:
- `results/metrics_summary.csv`
- `results/pca_train_data.png`
- `results/evaluation_curves.png` (ROC and PR Curves)
- `results/confusion_matrix_*.png`
- `results/decision_boundary_*.png`
