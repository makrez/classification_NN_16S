import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import itertools
import os
import torch

def plot_training_results(model, train_losses, test_losses, test_dataloader, device, label_encoder, directory):
    # Plot the training and test losses from the 11th epoch onward
    plt.figure()
    plt.plot(range(10, len(train_losses)), train_losses[10:], label="Training Loss")
    plt.plot(range(10, len(test_losses)), test_losses[10:], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(directory, 'loss_plot.png'))
    plt.show()

    # Get predictions for the test set
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in test_dataloader:
            sequence_data = batch["sequence"].permute(0, 2, 1).to(device)
            labels = batch["label"].to(device)

            outputs = model(sequence_data)
            _, preds = torch.max(outputs, 1)

            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = label_encoder.classes_

    # Plot confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=class_names)
    plt.savefig(os.path.join(directory, 'confusion_matrix_plot.png'))
    plt.show()


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
