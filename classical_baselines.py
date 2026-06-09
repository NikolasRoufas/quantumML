from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def train_classical_models(X_train, y_train, X_test, y_test):
    """Train SVM-RBF, Random Forest, and MLP baselines."""
    models = {
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale", random_state=42, probability=True
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            learning_rate_init=0.01,
            random_state=42,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"  Training {name}...", flush=True)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))
        print(f"    Train: {train_acc:.4f} | Test: {test_acc:.4f}", flush=True)
        results[name] = {
            "model": model,
            "train_acc": float(train_acc),
            "test_acc":  float(test_acc),
        }

    return results
