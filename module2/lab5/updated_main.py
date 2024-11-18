import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class DecisionTree:
    class Node:
        def __init__(self, attribute=None, outcome=None, branches=None):
            self.attribute = attribute
            self.outcome = outcome
            self.branches = branches or {}

    def __init__(self, max_depth=None, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        self.feature_subset = None

    @staticmethod
    def _calculate_entropy(labels):
        label_counts = Counter(labels)
        probabilities = [count / len(labels) for count in label_counts.values()]
        return -sum(p * np.log2(p) for p in probabilities)

    def _info_gain(self, data, labels, attribute):
        initial_entropy = self._calculate_entropy(labels)
        values = data[attribute].value_counts()
        weighted_entropy = 0

        for val in values.index:
            subset = labels[data[attribute] == val]
            weight = len(subset) / len(labels)
            weighted_entropy += weight * self._calculate_entropy(subset)

        return initial_entropy - weighted_entropy

    @staticmethod
    def _fill_missing_values(data):
        filled_data = data.copy()
        for col in filled_data.columns:
            filled_data[col] = filled_data[col].fillna(filled_data[col].mode()[0])
        return filled_data

    @staticmethod
    def _random_features(features):
        subset_size = int(np.sqrt(len(features)))
        return np.random.choice(features, size=subset_size, replace=False)

    def _build(self, data, labels, depth=0):
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(labels) < self.min_samples
            or len(set(labels)) == 1
        ):
            return self.Node(outcome=Counter(labels))

        best_attr, best_gain = None, -1
        for attr in data.columns:
            gain = self._info_gain(data, labels, attr)
            if gain > best_gain:
                best_attr, best_gain = attr, gain

        if best_gain <= 0:
            return self.Node(outcome=Counter(labels))

        node = self.Node(attribute=best_attr)
        for value in data[best_attr].unique():
            subset = data[best_attr] == value
            if subset.any():
                branch_data = data[subset].drop(columns=[best_attr])
                branch_labels = labels[subset]
                node.branches[value] = self._build(
                    branch_data, branch_labels, depth + 1
                )
        return node

    def train(self, features, labels):
        clean_data = self._fill_missing_values(features)
        self.feature_subset = self._random_features(clean_data.columns)
        reduced_data = clean_data[self.feature_subset]
        self.root = self._build(reduced_data, labels)

    def _predict_instance(self, instance, node):
        if node.outcome is not None:
            total = sum(node.outcome.values())
            probabilities = {k: v / total for k, v in node.outcome.items()}
            return probabilities

        attr_value = instance[node.attribute]
        if attr_value in node.branches:
            return self._predict_instance(instance, node.branches[attr_value])
        outcomes = Counter()
        for branch in node.branches.values():
            if branch.outcome is not None:
                outcomes.update(branch.outcome)

        total = sum(outcomes.values())
        probabilities = {k: v / total for k, v in outcomes.items()}
        return probabilities

    def predict(self, features):
        clean_data = self._fill_missing_values(features)
        subset = clean_data[self.feature_subset]
        return [self._predict_instance(row, self.root) for _, row in subset.iterrows()]

    def evaluate(self, features, labels):
        probabilities = self.predict(features)
        y_scores = np.array([prob.get("p", 0) for prob in probabilities])
        y_true = np.array([1 if label == "p" else 0 for label in labels])
        thresholds = np.linspace(0, 1, 100)
        y_pred = (y_scores >= 0.5).astype(int)
        accuracy = np.mean(y_pred == y_true)
        tp = sum((y_true == 1) & (y_pred == 1))
        fp = sum((y_true == 0) & (y_pred == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        fn = sum((y_true == 1) & (y_pred == 0))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        tpr, fpr = [], []
        for thresh in thresholds:
            pred_bin = (y_scores >= thresh).astype(int)
            tp = sum((y_true == 1) & (pred_bin == 1))
            tn = sum((y_true == 0) & (pred_bin == 0))
            fp = sum((y_true == 0) & (pred_bin == 1))
            fn = sum((y_true == 1) & (pred_bin == 0))
            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

        plt.figure(figsize=(8, 6))
        plt.plot(
            [0, 1], [0, 1], label="Random Classifier", color="gray", linestyle="--"
        )
        plt.plot(fpr + [0], tpr + [0], color="blue", label="ROC Curve")
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("AUC-ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

        # AUC-PR Calculation
        precision, recall = [], []

        for thresh in thresholds:
            pred_bin = (y_scores >= thresh).astype(int)
            tp = sum((y_true == 1) & (pred_bin == 1))
            fp = sum((y_true == 0) & (pred_bin == 1))
            fn = sum((y_true == 1) & (pred_bin == 0))
            precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

        recall.append(0)
        precision.append(1)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="green", label="Precision-Recall Curve")
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("AUC-PR Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


data = pd.read_csv("./module2/lab5/mushroom.csv")
features = data.drop(columns=["poisonous"])
labels = data["poisonous"]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
tree = DecisionTree(max_depth=5)
tree.train(X_train, y_train)
tree.evaluate(X_test, y_test)
