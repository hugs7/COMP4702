"""
Helper function to calculate and display the confusion matrix
"""


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def conf_matrix(y_true, y_pred, classes):
    """
    Calculate the confusion matrix for the given data and
    display it using ConfusionMatrixDisplay
    """

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)
    conf_matrix_disp = ConfusionMatrixDisplay(
        conf_matrix, display_labels=classes)
    conf_matrix_disp.plot(cmap="Blues")
    plt.show()

    return conf_matrix
