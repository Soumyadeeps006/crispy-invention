import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# Apply clean, modern styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 16,
    "legend.fontsize": 11,
})

def plot_pca_data(X, y, title="PCA Scatter Plot", save_path=None):
    """
    Plots the first two principal components of the data, highlighting Fraud and Legitimate cases.
    """
    plt.figure(figsize=(9, 7))
    
    # Class mapping for labels
    classes = ["Legitimate", "Fraud"]
    colors = ["#4C72B0", "#DD8452"]  # Elegant blue and coral/orange
    
    for class_val, class_name, color in zip([0, 1], classes, colors):
        idx = (y == class_val)
        plt.scatter(
            X[idx, 0], X[idx, 1],
            label=class_name,
            alpha=0.75,
            edgecolors='w',
            linewidths=0.5,
            s=60,
            color=color
        )
        
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title, pad=15)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved PCA plot to {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plots the confusion matrix using a beautiful Seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.ylabel("Actual Label", labelpad=10)
    plt.xlabel("Predicted Label", labelpad=10)
    plt.title(f"Confusion Matrix - {model_name}", pad=15)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved confusion matrix to {save_path}")
    plt.close()

def plot_roc_pr_curves(results_dict, y_true, save_dir="results"):
    """
    Plots ROC and Precision-Recall Curves for multiple models side-by-side.
    results_dict format: { 'Model Name': y_prob }
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Custom color palette for models
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    
    # 1. ROC Curve
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    for (model_name, y_prob), color in zip(results_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", lw=2, color=color)
        
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax1.legend(loc="lower right", frameon=True, facecolor="white")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    # 2. Precision-Recall Curve
    # Baseline is the ratio of fraud cases
    baseline = np.sum(y_true == 1) / len(y_true)
    ax2.plot([0, 1], [baseline, baseline], 'k--', alpha=0.5, label=f"No Skill ({baseline:.3f})")
    
    for (model_name, y_prob), color in zip(results_dict.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, label=f"{model_name} (PR-AUC = {pr_auc:.3f})", lw=2, color=color)
        
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="lower left", frameon=True, facecolor="white")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    
    plt.suptitle("Model Evaluation Curves", y=0.98)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "evaluation_curves.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved evaluation curves to {save_path}")
    plt.close()

def plot_decision_boundary_2d(model, X, y, title="Decision Boundary", save_path=None):
    """
    Plots the decision boundary of a model on the first two dimensions, 
    fixing all other dimensions (if > 2) to their mean values (which is 0 for PCA).
    """
    plt.figure(figsize=(10, 8))
    
    # Get range for first two principal components
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Create a grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                         np.linspace(y_min, y_max, 30))
    
    # Prepare grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    n_features = X.shape[1]
    
    if n_features > 2:
        # Pad with zeros (means of remaining PCA components)
        padding = np.zeros((grid_points.shape[0], n_features - 2))
        grid_points = np.hstack([grid_points, padding])
        
    # Evaluate decision function on grid
    if hasattr(model, "decision_function"):
        Z = model.decision_function(grid_points)
    else:
        # Fallback to probability of fraud (class 1)
        Z = model.predict_proba(grid_points)[:, 1]
        
    Z = Z.reshape(xx.shape)
    
    # Draw decision boundary and contours
    contour = plt.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.3)
    plt.colorbar(contour, label="Decision Function / Probability")
    
    # Plot boundary line (where decision function is 0, or prob is 0.5)
    # Check if we have positive/negative boundary or probability [0, 1]
    level = 0.5 if not hasattr(model, "decision_function") else 0.0
    plt.contour(xx, yy, Z, levels=[level], linewidths=2, colors='k')
    
    # Scatter plot data points
    classes = ["Legitimate", "Fraud"]
    colors = ["#4C72B0", "#DD8452"]
    for class_val, class_name, color in zip([0, 1], classes, colors):
        idx = (y == class_val)
        plt.scatter(
            X[idx, 0], X[idx, 1],
            label=class_name,
            alpha=0.8,
            edgecolors='w',
            linewidths=0.5,
            s=50,
            color=color
        )
        
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title, pad=15)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved decision boundary plot to {save_path}")
    plt.close()
