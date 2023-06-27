import os
import pickle
import yaml
from sklearn.model_selection import ParameterGrid
from model_summarise_functions import plot_confusion_matrix, plot_train_test_curves, print_f1_and_classification_report
import numpy as np

def summarize_model(model_evaluation_dir):
    with open('/scratch/mk_cas/datasets/train/label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    print(label_map)

    # Load the predictions and losses
    train_losses = np.load(os.path.join(model_evaluation_dir, 'train_losses.npy'))
    test_losses = np.load(os.path.join(model_evaluation_dir, 'test_losses.npy'))
    y_true = np.load(os.path.join(model_evaluation_dir, 'y_true.npy'))
    print(y_true)
    y_pred = np.load(os.path.join(model_evaluation_dir, 'y_pred.npy'))
    print(y_pred)

    plot_confusion_matrix(y_true, y_pred, label_map, model_evaluation_dir)
    plot_train_test_curves(train_losses, test_losses, model_evaluation_dir)
    print_f1_and_classification_report(y_true, y_pred, label_map, model_evaluation_dir)

# Load the configuration file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get the hyperparameters from the config file
lr_list = config['lr']
n_epoch_list = config['n_epoch']

# Define the hyperparameters
param_grid = {
    'lr': lr_list,
    'n_epoch': n_epoch_list
}

grid = ParameterGrid(param_grid)

# Loop over all combinations of parameters
for params in grid:
    # Construct the directory name from the parameters
    directory = f"lr={params['lr']}_n_epoch={params['n_epoch']}"
    # Call the summarize_model function
    summarize_model(os.path.join(directory, 'model_evaluation'))

