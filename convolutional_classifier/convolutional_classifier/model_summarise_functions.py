import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score
import seaborn as sns
import numpy as np
import pickle
import os

def plot_confusion_matrix(y_true, y_pred, labels_map, subdirectory):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    labels_map =  {v: k for k, v in labels_map.items()}

    # Create a heatmap using the confusion matrix
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", square=True, cbar=False,
                xticklabels=[labels_map[i] for i in np.unique(y_true)],
                yticklabels=[labels_map[i] for i in np.unique(y_true)], ax=ax)

    # Set axis labels and title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    # Rotate the tick labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
 
    # Save the figure
    plt.savefig(os.path.join(subdirectory, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')



def print_f1_and_classification_report(y_true, y_pred, labels_map, subdirectory):
    labels = sorted(list(labels_map.keys()), key=lambda x: labels_map[x])
    classification_rep = classification_report(y_true, y_pred, target_names=labels)
    f1_score_result = f1_score(y_true, y_pred, average='macro')

    # print and save to file
    with open(os.path.join(subdirectory, 'f1_classification_report.txt'), 'w') as f:
        print("Classification Report:", file=f)
        print(classification_rep, file=f)
        print("F1 Score:", file=f)
        print(f1_score_result, file=f)

    return classification_rep, f1_score_result

def plot_train_test_curves(train_losses, test_losses, subdirectory):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, test_losses, 'b', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Save the figure
    plt.savefig(os.path.join(subdirectory, 'train_test_curves.png'), dpi=300, bbox_inches='tight')



