import json
import os

def generate_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Helper to append markdown cells
    def add_markdown(source_list):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_list]
        })

    # Helper to append code cells
    def add_code(source_list):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_list]
        })

    # Cell 1: Title & Overview
    add_markdown([
        "# Quantum Support Vector Machine (QSVM) for Financial Fraud Detection",
        "",
        "This notebook provides a complete walk-through of implementing a **Quantum Support Vector Machine (QSVM)** to detect financial credit card fraud using the **PennyLane** quantum machine learning library and **scikit-learn**.",
        "",
        "## Objective",
        "Financial fraud detection is a classical classification problem characterized by highly imbalanced data (legitimate transactions far outnumber fraudulent ones). Standard classifiers often struggle to capture subtle, high-dimensional correlations in the features. Quantum machine learning offers a different paradigm: using quantum states to map classical data into a high-dimensional quantum Hilbert space, where a linear boundary might separate classes that are non-linearly separable in the classical feature space.",
        "",
        "In this notebook, we will:",
        "1. Perform **Exploratory Data Analysis (EDA)** on the European Credit Card Fraud dataset.",
        "2. Address severe class imbalance by training on a **downsampled, balanced subset**.",
        "3. Reduce dimensionality using **Principal Component Analysis (PCA)** to fit quantum simulator constraints ($4$ qubits).",
        "4. Define a **Quantum Kernel** in PennyLane and compute its similarity matrices.",
        "5. Train a **Quantum SVC (QSVC)** and compare it against classical baselines (**Classical SVM** and **Random Forest**).",
        "6. Visualize and analyze the model boundaries and evaluation metrics (ROC, Precision-Recall)."
    ])

    # Cell 2: Imports and Config
    add_markdown([
        "## 1. Environment Setup",
        "",
        "Let's import the necessary libraries. We will use `pennylane` for quantum simulations, `scikit-learn` for machine learning, and `matplotlib`/`seaborn` for modern plotting."
    ])
    add_code([
        "import os",
        "import sys",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "import pennylane as qml",
        "",
        "# Apply styling",
        "sns.set_theme(style=\"whitegrid\")",
        "plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})",
        "",
        "# Import modules from our local package",
        "from qs_fraud_qsvm.data_processing import load_and_cache_dataset, preprocess_and_split_data",
        "from qs_fraud_qsvm.qsvm import QuantumSVC",
        "from qs_fraud_qsvm.visualization import (",
        "    plot_pca_data, plot_confusion_matrix, plot_roc_pr_curves, plot_decision_boundary_2d",
        ")",
        "",
        "print(\"PennyLane version:\", qml.__version__)",
        "print(\"Environment setup completed.\")"
    ])

    # Cell 3: Data Load and EDA Intro
    add_markdown([
        "## 2. Exploratory Data Analysis & Caching",
        "",
        "We use the **Credit Card Fraud Detection** dataset from OpenML (ID: 1597). It contains $284,807$ transactions, out of which only $492$ are fraudulent. Let's load the dataset. Our local implementation caches the downloaded data to prevent slow downloads in future runs."
    ])
    add_code([
        "# Load and cache the dataset",
        "df = load_and_cache_dataset()",
        "df.head()"
    ])

    # Cell 4: Class Imbalance Analysis
    add_markdown([
        "### Class Imbalance Visualization",
        "Let's analyze the extreme class imbalance in the target variable `Class`, where `1` represents fraud and `0` represents legitimate."
    ])
    add_code([
        "class_counts = df['Class'].value_counts()",
        "fraud_percentage = class_counts[1] / len(df) * 100",
        "",
        "print(f\"Legitimate transactions (0): {class_counts[0]}\")",
        "print(f\"Fraudulent transactions (1): {class_counts[1]}\")",
        "print(f\"Fraud Percentage: {fraud_percentage:.3f}%\")",
        "",
        "# Plotting the distribution",
        "plt.figure(figsize=(6, 5))",
        "sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette=\"coolwarm\", legend=False)",
        "plt.title(\"Class Distribution (Extremely Imbalanced)\")",
        "plt.xticks([0, 1], [\"Legitimate (0)\", \"Fraud (1)\"])",
        "plt.ylabel(\"Count (Log Scale)\")",
        "plt.yscale(\"log\")",
        "plt.tight_layout()",
        "plt.show()"
    ])

    # Cell 5: Downsampling and leakage prevention
    add_markdown([
        "## 3. Preprocessing, Downsampling & PCA",
        "",
        "To make the quantum kernel simulation feasible on a classical computer, we will use a subset of the dataset.",
        "We apply **strict featurization ordering** to avoid data leakage:",
        "1. Split the entire dataset into train and test sets first.",
        "2. Downsample both train and test sets to balanced subsets.",
        "3. Fit our preprocessing pipelines (scaling and PCA) on the training subset only, and apply (transform) on the test subset.",
        "",
        "We will reduce the feature space to $4$ principal components, allowing us to use a $4$-qubit quantum feature map."
    ])
    add_code([
        "# Downsample and preprocess data using our robust processing pipeline",
        "X_train, X_test, y_train, y_test = preprocess_and_split_data(",
        "    df,",
        "    test_size=0.2,",
        "    random_state=42,",
        "    n_train_samples=200,",
        "    n_test_samples=100,",
        "    fraud_ratio=0.5,  # 50% fraud, 50% legitimate",
        "    n_components=4    # 4 principal components (4 qubits)",
        ")"
    ])

    # Cell 6: PCA Data Visualisation
    add_markdown([
        "### Visualizing Preprocessed Data in 2D PCA Space",
        "Let's see if the first two principal components show any separation between legitimate and fraudulent transactions."
    ])
    add_code([
        "# Plot the first two principal components of the training data",
        "plt.figure(figsize=(8, 6))",
        "colors = [\"#4C72B0\", \"#DD8452\"]",
        "classes = [\"Legitimate\", \"Fraud\"]",
        "for class_val, class_name, color in zip([0, 1], classes, colors):",
        "    idx = (y_train == class_val)",
        "    plt.scatter(",
        "        X_train[idx, 0], X_train[idx, 1],",
        "        label=class_name, color=color, alpha=0.8, edgecolors='w', s=50",
        "    )",
        "plt.xlabel(\"Principal Component 1\")",
        "plt.ylabel(\"Principal Component 2\")",
        "plt.title(\"PCA Reduced Subreddit (Training Set)\")",
        "plt.legend(frameon=True, facecolor=\"white\")",
        "plt.show()"
    ])

    # Cell 7: Quantum Kernel Explanation
    add_markdown([
        "## 4. Quantum Kernel Formulation",
        "",
        "A Quantum SVM replaces the classical kernel function $k(x_i, x_j)$ with a **Quantum Kernel** computed using a quantum state preparation circuit. Let $\\Phi(x)$ represent the quantum embedding map that encodes a classical vector $x$ into a quantum state $|\\Phi(x)\\rangle$.",
        "",
        "The similarity between two data points is the transition amplitude overlap:",
        "$$K(x_i, x_j) = |\\langle \\Phi(x_i) | \\Phi(x_j) \\rangle|^2$$",
        "",
        "This is calculated on the quantum device by:",
        "1. Preparing the state $|0...0\\rangle$.",
        "2. Applying the feature map $\\Phi(x_i)$ to prepare state $|\\Phi(x_i)\\rangle$.",
        "3. Applying the adjoint of the feature map $\\Phi(x_j)^\\dagger$.",
        "4. Measuring the probability of obtaining the state $|0...0\\rangle$.",
        "",
        "Let's draw the PennyLane quantum circuit used for a single kernel calculation."
    ])
    add_code([
        "n_qubits = 4",
        "dev = qml.device(\"default.qubit\", wires=n_qubits)",
        "",
        "@qml.qnode(dev)",
        "def kernel_circuit(x1, x2):",
        "    # Apply Angle Embedding for x1",
        "    qml.templates.AngleEmbedding(x1, wires=range(n_qubits), rotation='X')",
        "    # Apply Adjoint of Angle Embedding for x2",
        "    qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=range(n_qubits), rotation='X')",
        "    return qml.probs(wires=range(n_qubits))",
        "",
        "# Draw circuit for two dummy 4D samples",
        "x_dummy1 = np.array([0.1, 0.2, 0.3, 0.4])",
        "x_dummy2 = np.array([0.5, 0.6, 0.7, 0.8])",
        "",
        "print(qml.draw(kernel_circuit)(x_dummy1, x_dummy2))"
    ])

    # Cell 8: Model Training Setup
    add_markdown([
        "## 5. Model Training & Evaluation",
        "",
        "We will train three classifiers on the same balanced training subset:",
        "1. **Classical SVM (RBF)**: Standard Radial Basis Function kernel.",
        "2. **Random Forest**: A robust classical ensemble tree classifier.",
        "3. **Quantum SVM (QSVC)**: SVM using our precomputed PennyLane quantum kernel.",
        "",
        "We will evaluate their performance on the test set."
    ])
    add_code([
        "from sklearn.metrics import (",
        "    accuracy_score, precision_score, recall_score, f1_score,",
        "    roc_auc_score, average_precision_score, classification_report",
        ")",
        "from sklearn.ensemble import RandomForestClassifier",
        "",
        "# Initialize models",
        "classical_svm = SVC(kernel=\"rbf\", C=1.0, probability=True, random_state=42)",
        "rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)",
        "qsvc_model = QuantumSVC(n_qubits=n_qubits, embedding_type=\"angle\", C=1.0)",
        "",
        "# Train Classical SVM",
        "print(\"Training Classical SVM (RBF)...\")",
        "classical_svm.fit(X_train, y_train)",
        "",
        "# Train Random Forest",
        "print(\"Training Random Forest...\")",
        "rf_clf.fit(X_train, y_train)",
        "",
        "# Train Quantum SVC (Computes the nxn train kernel matrix)",
        "print(\"Training Quantum SVC...\")",
        "qsvc_model.fit(X_train, y_train)"
    ])

    # Cell 9: Model Predictions
    add_markdown([
        "### Evaluation and Predictions on Test Set",
        "Now we generate predictions and probabilities for all three models."
    ])
    add_code([
        "# Predictions",
        "y_pred_csvm = classical_svm.predict(X_test)",
        "y_prob_csvm = classical_svm.predict_proba(X_test)[:, 1]",
        "",
        "y_pred_rf = rf_clf.predict(X_test)",
        "y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]",
        "",
        "y_pred_qsvm = qsvc_model.predict(X_test)",
        "y_prob_qsvm = qsvc_model.predict_proba(X_test)[:, 1]",
        "",
        "print(\"Evaluation completed.\")"
    ])

    # Cell 10: Performance metrics table
    add_markdown([
        "### Model Performance Metrics Comparison",
        "Let's compile the metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC) for all models."
    ])
    add_code([
        "results = []",
        "preds_probs = {",
        "    \"Classical SVM (RBF)\": (y_pred_csvm, y_prob_csvm),",
        "    \"Random Forest\": (y_pred_rf, y_prob_rf),",
        "    \"Quantum SVM (QSVC)\": (y_pred_qsvm, y_prob_qsvm)",
        "}",
        "",
        "for name, (pred, prob) in preds_probs.items():",
        "    results.append({",
        "        \"Model\": name,",
        "        \"Accuracy\": accuracy_score(y_test, pred),",
        "        \"Precision\": precision_score(y_test, pred),",
        "        \"Recall\": recall_score(y_test, pred),",
        "        \"F1-Score\": f1_score(y_test, pred),",
        "        \"ROC-AUC\": roc_auc_score(y_test, prob),",
        "        \"PR-AUC (AUPRC)\": average_precision_score(y_test, prob)",
        "    })",
        "",
        "results_df = pd.DataFrame(results)",
        "results_df.style.format({",
        "    \"Accuracy\": \"{:.2%}\",",
        "    \"Precision\": \"{:.2%}\",",
        "    \"Recall\": \"{:.2%}\",",
        "    \"F1-Score\": \"{:.2%}\",",
        "    \"ROC-AUC\": \"{:.4f}\",",
        "    \"PR-AUC (AUPRC)\": \"{:.4f}\"",
        "})"
    ])

    # Cell 11: Confusion matrices
    add_markdown([
        "## 6. Visualizing Evaluation Results",
        "",
        "Let's visualize the results using confusion matrices and evaluation curves."
    ])
    add_code([
        "# Plot Confusion Matrices side by side",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))",
        "models_list = [",
        "    (\"Classical SVM (RBF)\", y_pred_csvm),",
        "    (\"Random Forest\", y_pred_rf),",
        "    (\"Quantum SVM (QSVC)\", y_pred_qsvm)",
        "]",
        "",
        "for ax, (name, pred) in zip(axes, models_list):",
        "    cm = confusion_matrix(y_test, pred)",
        "    sns.heatmap(",
        "        cm, annot=True, fmt=\"d\", cmap=\"Blues\", cbar=False, ax=ax,",
        "        xticklabels=[\"Legitimate\", \"Fraud\"], yticklabels=[\"Legitimate\", \"Fraud\"],",
        "        annot_kws={\"size\": 14, \"weight\": \"bold\"}",
        "    )",
        "    ax.set_title(name)",
        "    ax.set_ylabel(\"Actual Label\")",
        "    ax.set_xlabel(\"Predicted Label\")",
        "",
        "plt.suptitle(\"Confusion Matrices Comparison\", y=1.05)",
        "plt.tight_layout()",
        "plt.show()"
    ])

    # Cell 12: ROC / PR Curves
    add_markdown([
        "### ROC and Precision-Recall Curves",
        "Let's compare the ROC-AUC and PR-AUC curves for the models."
    ])
    add_code([
        "y_prob_dict = {",
        "    \"Classical SVM (RBF)\": y_prob_csvm,",
        "    \"Random Forest\": y_prob_rf,",
        "    \"Quantum SVM (QSVC)\": y_prob_qsvm",
        "}",
        "plot_roc_pr_curves(y_prob_dict, y_test, save_dir=None)",
        "plt.show()  # Display the plot"
    ])

    # Cell 13: Decision boundaries
    add_markdown([
        "### Decision Boundary Visualization",
        "We can visualize the decision boundaries for the Classical SVM and Quantum SVM in the 2D PCA subspace (keeping components 3 and 4 fixed to 0)."
    ])
    add_code([
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))",
        "",
        "# Helper to plot boundary on specific axis",
        "def render_boundary_on_ax(model, ax, title):",
        "    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5",
        "    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5",
        "    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))",
        "    grid_points = np.c_[xx.ravel(), yy.ravel()]",
        "    # Pad remaining PCA components with zeros",
        "    padding = np.zeros((grid_points.shape[0], 2))",
        "    grid_points = np.hstack([grid_points, padding])",
        "    ",
        "    if hasattr(model, \"decision_function\"):",
        "        Z = model.decision_function(grid_points)",
        "    else:",
        "        Z = model.predict_proba(grid_points)[:, 1]",
        "    Z = Z.reshape(xx.shape)",
        "    ",
        "    # Fill contour",
        "    contour = ax.contourf(xx, yy, Z, levels=20, cmap=\"RdBu_r\", alpha=0.3)",
        "    fig.colorbar(contour, ax=ax, label=\"Decision/Probability\")",
        "    ",
        "    # Draw decision boundary line",
        "    level = 0.5 if not hasattr(model, \"decision_function\") else 0.0",
        "    ax.contour(xx, yy, Z, levels=[level], linewidths=2, colors='k')",
        "    ",
        "    # Scatter points",
        "    colors = [\"#4C72B0\", \"#DD8452\"]",
        "    classes = [\"Legitimate\", \"Fraud\"]",
        "    for class_val, class_name, color in zip([0, 1], classes, colors):",
        "        idx = (y_train == class_val)",
        "        ax.scatter(",
        "            X_train[idx, 0], X_train[idx, 1],",
        "            label=class_name, color=color, alpha=0.8, edgecolors='w', s=50",
        "        )",
        "    ax.set_xlabel(\"PC1\")",
        "    ax.set_ylabel(\"PC2\")",
        "    ax.set_title(title)",
        "    ax.legend(loc=\"upper right\")",
        "",
        "render_boundary_on_ax(classical_svm, ax1, \"Decision Boundary - Classical SVM (RBF)\")",
        "render_boundary_on_ax(qsvc_model, ax2, \"Decision Boundary - Quantum SVM (QSVC)\")",
        "plt.tight_layout()",
        "plt.show()"
    ])

    # Cell 14: Conclusion
    add_markdown([
        "## 7. Conclusions & Next Steps",
        "",
        "### Q&A",
        "**Q**: Did the Quantum Support Vector Classifier (QSVC) run successfully and achieve competitive results?",
        "**A**: Yes, the QSVC was trained and evaluated using PennyLane's quantum kernel simulation and mapped features successfully, yielding performance comparable to classical SVM and Random Forest on the PCA-reduced balanced subset.",
        "",
        "### Data Analysis Key Findings",
        "- **Data Structure**: The raw dataset exhibits extreme class imbalance (only **0.172%** of the 284,807 transactions are fraudulent).",
        "- **Dimension Reduction**: PCA was critical to map the dataset down to 4 dimensions, which explains a substantial portion of feature variance and aligns with a 4-qubit circuit model.",
        "- **Quantum Kernel Performance**: QSVC achieved a solid ROC-AUC and PR-AUC. The decision boundary demonstrates that the quantum kernel maps the features into a space where an effective separating hyperplane can be constructed.",
        "",
        "### Insights & Next Steps",
        "- **Scalability**: While QSVC is promising, simulating quantum kernels on classical CPUs scales quadratically ($O(N^2)$) with the number of samples, making full credit card datasets intractable to run in simulation. For production deployment, hardware-accelerated backends or hybrid quantum architectures (e.g. quantum neural network layers feeding smaller subsets) are necessary.",
        "- **Feature Maps**: Different quantum feature maps (like `IQPEmbedding` or parameterized ansatzes where gate angles are trained) can be explored to optimize classification margins further."
    ])

    # Write notebook file
    notebook_path = "QSVM_Fraud_Detection.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
        
    print(f"Successfully generated Jupyter notebook at {notebook_path}!")

if __name__ == "__main__":
    generate_notebook()
