"""
Driver script for Neural Network model
06/05/2024
Hugo Burton
"""

from typing import List
import torch
from colorama import Fore, Style

import process_data
import results

from nn import train
from nn import nn_model

def run_nn_model(dataset_file_path: str, columns: List[str]) -> None:

    num_variables_predicting = 2

    # First num_variables_predicting columns are the variables we are predicting
    # These will be string labels which need to be flattened into a 1D array.
    # So the output dimension is sum of num classes over the num_variables_predicting
    # we are predicting
    y_labels = columns[0:num_variables_predicting]
    X_labels = columns[num_variables_predicting:]

    # --- Dataset ---

    # Read the dataset
    dataset, X_train, y_train, X_test, y_test, y_classes = (
        process_data.process_classification_data(
            dataset_file_path, X_labels, y_labels, 0.3
        )
    )

    print("Training data stats")
    print("Shape:", X_train.shape)
    print("Classes:", y_classes)

    #################

    classes_in_output_vars = [len(classes) for classes in y_classes]

    # Ouptut dimension is sum of classes in each output variable
    # because of one hot encoding
    dim_output = sum(classes_in_output_vars)
    dim_input = dataset.shape[1] - num_variables_predicting
    normalising_factor = 1.0
    hidden_layer_dims = [100, 150, 100]

    # Hyperparameters
    epochs = int(1e3)
    batch_size = 1000
    learning_rate = 1e-4

    # Convert to tensor
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float32)

    X_train = to_tensor(X_train)
    y_train = to_tensor(y_train)

    X_test = to_tensor(X_test)
    y_test = to_tensor(y_test)

    # ----- Device ------
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    print("Training data shape:", X_train.shape, "x", y_train.shape)
    print("Validation data shape:", X_test.shape, "x", y_test.shape)

    # Instantiate the model and move it to the specified device
    sequential_model = nn_model.create_sequential_model(
        dim_input, dim_output, hidden_layer_dims
    ).to(device)

    print(f"{Fore.LIGHTCYAN_EX}Model: \n{Style.RESET_ALL}{sequential_model}\n")

    # --- Loss Function ---
    # For classification problems, usually use cross entropy loss
    # For regression problems, usually use mean squared error

    # Both accuracy and loss are important

    # --- Optimiser ---

    # Stochastic gradient descent is the most popular optimiser when
    # training a neural network

    # We only compute the gradient on a subset of the data because this
    # is far faster than computing the gradient on the entire dataset.
    # Given a large enough subset, the gradient will be a good approximation
    # of the true gradient.

    # Classification problem so define cross entropy loss
    # and stochastic gradient descent optimiser

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    optimiser = torch.optim.SGD(sequential_model.parameters(), lr=learning_rate)

    # --- Training Loop ---

    metrics = []
    # for i in tqdm(range(int(epochs))):
    for i in range(int(epochs)):
        metrics = train.nn_train(
            i,
            X_train,
            y_train,
            batch_size,
            sequential_model,
            criterion,
            optimiser,
            epochs,
            metrics,
            classes_in_output_vars,
        )

    results.show_training_results(metrics)
