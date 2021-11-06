
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def plot_f1_scores(train_history):
    """ plotting f1 scores given the train history """
    e_index = [i + 1 for i in range(len(train_history) - 1)]
    plt.figure()
    plt.title("F1-scores (1)")
    plt.xlabel("Epochs")
    plt.plot(e_index,
             [th[-2].item() for th in train_history[:-1]],
             label="Dev micro F1")
    plt.plot(e_index,
             [th[-1].item() for th in train_history[:-1]],
             label="Dev macro F1")
    plt.scatter([len(train_history) - 1],
                [train_history[-1][-2]],
                label="Test micro F1")
    plt.scatter([len(train_history) - 1],
                [train_history[-1][-1]],
                label="Test macro F1")
    plt.legend()
    plt.savefig("./figures/f1_scores_training.png")


def plot_samples_per_class(Y_tokenized, index_to_label):
    """ plotting number of samples per class in test data """
    Y_test_flattend = [item for sublist in Y_tokenized for item in sublist]
    num_samples_per_class = [0 for _ in range(len(index_to_label))]
    for label in Y_test_flattend:
        num_samples_per_class[label] += 1
    class_dict = dict()
    for i, num_samples in enumerate(num_samples_per_class):
        class_dict[index_to_label[i]] = num_samples
    plt.figure()
    plt.title("Number of samples per class in the test dataset (2)")
    plt.bar(class_dict.keys(), class_dict.values())
    plt.savefig("./figures/samples_per_class.png")


def plot_confusion(confusion_matrix, index_to_label):
    """ plotting confusion matrix and most common errors """
    fig, ax = plt.subplots()
    hot = cm.get_cmap('hot', 512)
    newcmp = ListedColormap(hot(np.geomspace(0.65, 1e-5, 1000)))
    matrix = ax.imshow(confusion_matrix, cmap=newcmp)
    ax.set_xticks(range(len(confusion_matrix)))
    ax.set_yticks(range(len(confusion_matrix)))
    ax.set_xticklabels(index_to_label)
    ax.set_yticklabels(index_to_label)
    plt.setp(ax.get_xticklabels(), rotation=30,
             ha="right", rotation_mode="anchor")
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            _ = ax.text(
                j, i, confusion_matrix[i][j], ha="center", va="center", color="w")
    fig.colorbar(matrix, ax=ax)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.ylabel("Target Class")
    plt.xlabel("Output Class")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title("Confusion Matrixs (3)")
    plt.savefig("./figures/confusion_heatmap.png")

    errors = dict()
    error_1 = 0
    error_2 = 0
    error_3 = 0
    for i, row in enumerate(confusion_matrix):
        for j, el in enumerate(row):
            if i == j:
                continue
            if index_to_label[i] == "O":
                error_1 += el
            elif index_to_label[j] == "O":
                error_2 += el
            else:
                error_3 += el
            errors[f"{index_to_label[i]} but {index_to_label[j]}"] = el
    errors_by_amount = sorted(errors, key=errors.get, reverse=True)
    n_most_common_errors = 10
    fig, ax = plt.subplots()
    most_commen_errors = [errors[name]
                          for name in errors_by_amount[:n_most_common_errors]]
    most_commen_error = max([max(row) for row in confusion_matrix])
    most_commen_error_colors = [newcmp(v / most_commen_error) for v in most_commen_errors]
    plt.bar(errors_by_amount[:n_most_common_errors],
            most_commen_errors, color=most_commen_error_colors)
    plt.setp(ax.get_xticklabels(), rotation=30,
             ha="right", rotation_mode="anchor")
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.xlabel("<Target> but <Outpu> sorted by frequency")
    plt.ylabel("Frequency")
    plt.title("Error Frequency (4)")
    plt.savefig("./figures/error_frequency.png")

    fig, ax = plt.subplots()
    types_of_errors = ["O but <Entity>", "<Entity> but O", "Wrong Entity"]
    value_types_of_errors = [error_1, error_2, error_3]
    plt.bar(types_of_errors, value_types_of_errors)
    plt.setp(ax.get_xticklabels(), rotation=10,
             ha="right", rotation_mode="anchor")
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.xlabel("Kind of Error")
    plt.ylabel("Frequency")
    plt.title("Three Types of Errors (5)")
    plt.savefig("./figures/three_types_of_error.png")


def plot_evaluated_buckets(buckets):
    for i, (bucket_name, bucket) in enumerate(buckets.items()):
        plt.figure()
        plt.title(
            f"Macro F1-scores for differences in {bucket_name} ({i + 6})")
        labels = list()
        data = list()
        for macro_f1, subbucket_name in bucket:
            labels.append(subbucket_name)
            data.append(macro_f1)
        plt.bar(labels, data)
        plt.xticks(rotation=15)
        plt.ylabel("Macro F1-score")
        plt.gcf().subplots_adjust(bottom=0.25)
        plt.savefig("./figures/" + bucket_name.replace(" ", "_") +
                    ".png", facecolor='white', transparent=False)
