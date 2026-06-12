import os
import argparse
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# Import package modules
from qs_fraud_qsvm.data_processing import load_and_cache_dataset, preprocess_and_split_data
from qs_fraud_qsvm.qsvm import QuantumSVC
from qs_fraud_qsvm.visualization import (
    plot_pca_data, plot_confusion_matrix, plot_roc_pr_curves, plot_decision_boundary_2d
)

def run_pipeline(n_train=200, n_test=100, fraud_ratio=0.5, n_qubits=4, C=1.0, embedding_type="angle"):
    print("="*60)
    print("    QUANTUM SVM FINANCIAL FRAUD DETECTION PIPELINE")
    print("="*60)
    print(f"Parameters:\n- Train samples: {n_train}\n- Test samples: {n_test}")
    print(f"- Fraud ratio: {fraud_ratio}\n- Qubits (PCA dimensions): {n_qubits}")
    print(f"- SVM C parameter: {C}\n- Embedding: {embedding_type}")
    print("="*60)
    
    # 1. Load and Cache data
    df = load_and_cache_dataset()
    
    # 2. Preprocess, downsample and split
    X_train, X_test, y_train, y_test = preprocess_and_split_data(
        df,
        n_train_samples=n_train,
        n_test_samples=n_test,
        fraud_ratio=fraud_ratio,
        n_components=n_qubits
    )
    
    # Save a plot of the PCA training data
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    plot_pca_data(X_train, y_train, title="Training Data (First 2 Principal Components)", save_path=os.path.join(results_dir, "pca_train_data.png"))
    plot_pca_data(X_test, y_test, title="Test Data (First 2 Principal Components)", save_path=os.path.join(results_dir, "pca_test_data.png"))

    # 3. Initialize models
    print("\nInitializing models...")
    models = {
        "Classical SVM (RBF)": SVC(kernel="rbf", C=C, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Quantum SVM (QSVC)": QuantumSVC(n_qubits=n_qubits, embedding_type=embedding_type, C=C)
    }
    
    # Train and evaluate models
    metrics = []
    y_prob_dict = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        if name == "Quantum SVM (QSVC)":
            # Fits and computes precomputed quantum kernel
            model.fit(X_train, y_train)
            print(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            model.fit(X_train, y_train)
            print(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        y_prob_dict[name] = y_prob
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        metrics.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC (AUPRC)": pr_auc
        })
        
        # Plot confusion matrix
        plot_confusion_matrix(
            y_test, y_pred,
            model_name=name,
            save_path=os.path.join(results_dir, f"confusion_matrix_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png")
        )
        
        # Plot decision boundaries (only for SVMs)
        if "SVM" in name:
            boundary_name = f"decision_boundary_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
            plot_decision_boundary_2d(
                model, X_train, y_train,
                title=f"Decision Boundary - {name}",
                save_path=os.path.join(results_dir, boundary_name)
            )
            
    # Plot ROC and PR Curves side-by-side
    plot_roc_pr_curves(y_prob_dict, y_test, save_dir=results_dir)
    
    # 4. Compile metrics into a DataFrame
    metrics_df = pd.DataFrame(metrics)
    print("\n" + "="*80)
    print("                          MODEL EVALUATION SUMMARY")
    print("="*80)
    print(metrics_df.to_string(index=False, formatters={
        "Accuracy": "{:.2%}".format,
        "Precision": "{:.2%}".format,
        "Recall": "{:.2%}".format,
        "F1-Score": "{:.2%}".format,
        "ROC-AUC": "{:.4f}".format,
        "PR-AUC (AUPRC)": "{:.4f}".format,
    }))
    print("="*80)
    
    # Save metrics to CSV
    metrics_path = os.path.join(results_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics summary to {metrics_path}")
    print("\nAll visualizations have been saved in the 'results/' folder.")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the QSVM Fraud Detection Pipeline")
    parser.add_argument("--n_train", type=int, default=200, help="Number of training samples")
    parser.add_argument("--n_test", type=int, default=100, help="Number of test samples")
    parser.add_argument("--fraud_ratio", type=float, default=0.5, help="Ratio of fraud cases in subsets")
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits (PCA dimensions)")
    parser.add_argument("--C", type=float, default=1.0, help="SVM regularization parameter")
    parser.add_argument("--embedding", type=str, default="angle", choices=["angle", "iqp"], help="Quantum embedding type")
    
    args = parser.parse_args()
    
    run_pipeline(
        n_train=args.n_train,
        n_test=args.n_test,
        fraud_ratio=args.fraud_ratio,
        n_qubits=args.n_qubits,
        C=args.C,
        embedding_type=args.embedding
    )
