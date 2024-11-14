import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from collections import Counter


def load_data(filepath):
    data = []
    labels = []
    with open(filepath, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            labels.append(1 if row[0] == "p" else 0)  # Poisonous = 1, Edible = 0
            data.append(row[1:])
    return np.array(data), np.array(labels)


def encode_features(data):
    encoded_data = []
    for col in data.T:
        unique_vals = {val: idx for idx, val in enumerate(set(col))}
        encoded_data.append([unique_vals[val] for val in col])
    return np.array(encoded_data).T


data, labels = load_data("mushroom.csv")
data = encode_features(data)


def select_features(data):
    n_features = data.shape[1]
    num_features_to_select = int(np.sqrt(n_features))
    selected_indices = random.sample(range(n_features), num_features_to_select)
    return data[:, selected_indices], selected_indices


data, selected_features = select_features(data)


def gini_impurity(labels):
    counts = Counter(labels)
    impurity = 1.0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / len(labels)
        impurity -= prob_of_lbl**2
    return impurity


def split_data(data, labels, feature, threshold):
    left_mask = data[:, feature] == threshold
    right_mask = ~left_mask
    return (data[left_mask], labels[left_mask]), (data[right_mask], labels[right_mask])


def find_best_split(data, labels):
    best_gini = 1
    best_split = None
    for feature in range(data.shape[1]):
        for threshold in set(data[:, feature]):
            (left_data, left_labels), (right_data, right_labels) = split_data(
                data, labels, feature, threshold
            )
            gini = (len(left_labels) / len(labels)) * gini_impurity(left_labels) + (
                len(right_labels) / len(labels)
            ) * gini_impurity(right_labels)
            if gini < best_gini:
                best_gini = gini
                best_split = (feature, threshold)
    return best_split


class TreeNode:
    def __init__(self, gini, num_samples, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.predicted_class = predicted_class
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None


def build_tree(data, labels, depth=0, max_depth=5):
    num_samples_per_class = Counter(labels)
    predicted_class = max(num_samples_per_class, key=num_samples_per_class.get)
    node = TreeNode(
        gini=gini_impurity(labels),
        num_samples=len(labels),
        predicted_class=predicted_class,
    )

    if depth >= max_depth or node.gini == 0:
        return node

    feature, threshold = find_best_split(data, labels)
    if feature is None:
        return node

    node.feature = feature
    node.threshold = threshold
    (left_data, left_labels), (right_data, right_labels) = split_data(
        data, labels, feature, threshold
    )
    node.left = build_tree(left_data, left_labels, depth + 1, max_depth)
    node.right = build_tree(right_data, right_labels, depth + 1, max_depth)
    return node


def predict(sample, tree):
    if tree.left is None and tree.right is None:  # Leaf node
        return tree.predicted_class
    if sample[tree.feature] == tree.threshold:
        return predict(sample, tree.left)
    else:
        return predict(sample, tree.right)


def predict_all(data, tree):
    return np.array([predict(sample, tree) for sample in data])


def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return accuracy, precision, recall


tree = build_tree(data, labels, max_depth=5)
predictions = predict_all(data, tree)
accuracy, precision, recall = calculate_metrics(labels, predictions)

print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")


def calculate_roc_curve(y_true, y_scores):
    thresholds = np.linspace(0, 1, num=100)
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list


# Plot ROC curve
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, marker=".")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.show()


# Assuming y_scores is generated as the confidence level (0 or 1 in this binary case)
y_scores = predictions  # For simplicity, let's assume predictions as confidence here
fpr, tpr = calculate_roc_curve(labels, y_scores)
plot_roc_curve(fpr, tpr)


# PR curve
def calculate_pr_curve(y_true, y_scores):
    thresholds = np.linspace(0, 1, num=100)
    precision_list = []
    recall_list = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    return recall_list, precision_list


# Plot PR curve
def plot_pr_curve(recall, precision):
    plt.plot(recall, precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()


# Calculate and plot PR curve
precision, recall = calculate_pr_curve(labels, y_scores)
plot_pr_curve(recall, precision)
