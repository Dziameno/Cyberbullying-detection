def plot_matrix(pp, pn, nnp, nn, save_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # Confusion matrix
    cm = np.array([[pp, pn], [nnp, nn]])

    # Confusion matrix plot
    sns.heatmap(cm, annot=True, fmt="d", cmap="binary", xticklabels=["Non-harmful", "Harmful"],
                yticklabels=["Non-harmful", "Harmful"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.savefig(save_dir)
    plt.show()