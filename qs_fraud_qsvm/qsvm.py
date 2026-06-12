import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from tqdm import tqdm

class QuantumSVC:
    """
    Quantum Support Vector Classifier (QSVC) using PennyLane for quantum kernel computation
    and scikit-learn's SVC for training and prediction.
    """
    def __init__(self, n_qubits, embedding_type="angle", C=1.0, random_state=42):
        self.n_qubits = n_qubits
        self.embedding_type = embedding_type
        self.C = C
        self.random_state = random_state
        
        # Initialize the PennyLane device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Define the QNode for the kernel
        self.qnode = qml.QNode(self._kernel_circuit, self.dev)
        
        # Placeholder for the underlying SVC and training data
        self.clf = SVC(kernel="precomputed", C=self.C, probability=True, random_state=self.random_state)
        self.X_train = None

    def _kernel_circuit(self, x1, x2):
        """
        Quantum circuit to evaluate the overlap |<phi(x1)|phi(x2)>|^2.
        """
        # Embed the first data point
        if self.embedding_type == "angle":
            qml.templates.AngleEmbedding(x1, wires=range(self.n_qubits), rotation="X")
        elif self.embedding_type == "iqp":
            qml.templates.IQPEmbedding(x1, wires=range(self.n_qubits), n_repeats=1)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
            
        # Apply the adjoint (inverse) of the embedding for the second data point
        if self.embedding_type == "angle":
            qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=range(self.n_qubits), rotation="X")
        elif self.embedding_type == "iqp":
            qml.adjoint(qml.templates.IQPEmbedding)(x2, wires=range(self.n_qubits), n_repeats=1)
            
        # Return probability distribution; the first element (index 0) corresponds to the state |0...0>
        return qml.probs(wires=range(self.n_qubits))

    def compute_kernel_value(self, x1, x2):
        """
        Computes the quantum kernel value between two single samples.
        """
        # The |0...0> state probability is the transition amplitude overlap
        return self.qnode(x1, x2)[0]

    def compute_kernel_matrix(self, X1, X2, is_train=False):
        """
        Computes the quantum kernel matrix between two sets of samples X1 and X2.
        If is_train=True, exploits symmetry and diagonal constraints to speed up computation.
        """
        n1 = len(X1)
        n2 = len(X2)
        
        matrix = np.zeros((n1, n2))
        
        if is_train:
            # X1 and X2 are the same training set, so the matrix is symmetric and diagonal is 1.0
            print(f"Computing symmetric training kernel matrix of shape ({n1}, {n1})...")
            # Set diagonal
            np.fill_diagonal(matrix, 1.0)
            
            # Compute upper triangle
            total_iterations = (n1 * (n1 - 1)) // 2
            with tqdm(total=total_iterations, desc="QSVM Train Kernel") as pbar:
                for i in range(n1):
                    for j in range(i + 1, n1):
                        val = self.compute_kernel_value(X1[i], X1[j])
                        matrix[i, j] = val
                        matrix[j, i] = val
                        pbar.update(1)
        else:
            # X1 (test) and X2 (train) are different, compute the full grid
            print(f"Computing test/evaluation kernel matrix of shape ({n1}, {n2})...")
            total_iterations = n1 * n2
            with tqdm(total=total_iterations, desc="QSVM Test Kernel") as pbar:
                for i in range(n1):
                    for j in range(n2):
                        matrix[i, j] = self.compute_kernel_value(X1[i], X2[j])
                        pbar.update(1)
                        
        return matrix

    def fit(self, X, y):
        """
        Fits the Quantum SVC model.
        """
        self.X_train = np.array(X)
        K_train = self.compute_kernel_matrix(self.X_train, self.X_train, is_train=True)
        print("Training classical Support Vector Machine with precomputed quantum kernel...")
        self.clf.fit(K_train, y)
        print("Model training complete.")
        return self

    def predict(self, X):
        """
        Predicts labels for X.
        """
        if self.X_train is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
        X_arr = np.array(X)
        K_test = self.compute_kernel_matrix(X_arr, self.X_train, is_train=False)
        return self.clf.predict(K_test)

    def predict_proba(self, X):
        """
        Predicts class probabilities for X.
        """
        if self.X_train is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
        X_arr = np.array(X)
        K_test = self.compute_kernel_matrix(X_arr, self.X_train, is_train=False)
        return self.clf.predict_proba(K_test)

    def decision_function(self, X):
        """
        Evaluates the decision function for X.
        """
        if self.X_train is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
        X_arr = np.array(X)
        K_test = self.compute_kernel_matrix(X_arr, self.X_train, is_train=False)
        return self.clf.decision_function(K_test)
