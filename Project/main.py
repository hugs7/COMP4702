"""
Main Driver file for project
"""

from tqdm import tqdm
from welcome import welcome
import train
import model
import torch
import os
from colorama import Fore, Style
import process_data
import results
from dataset import DATASET_MAPPING


def run_model(dataset_name: str) -> None:
    if dataset_name not in DATASET_MAPPING:
        raise ValueError(
            f"{Fore.RED}Dataset {dataset_name} not found{Style.RESET_ALL}")

    dataset_file_name, columns = DATASET_MAPPING[dataset_name]

    dim_output = 1  # Things we are predicting

    y_labels = columns[0:dim_output]
    X_labels = columns[dim_output:]

    # --- Dataset ---

    folder_of_script = os.path.dirname(__file__)
    data_folder = os.path.join(folder_of_script, "data")

    # Create data folder if it does not exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    model_data_path = os.path.join(data_folder, dataset_file_name)

    # Read the dataset
    dataset, X_train, y_train, X_test, y_test = (
        process_data.process_classification_data(
            model_data_path, X_labels, y_labels, 0.3
        )
    )

    print("Training data stats")
    print("Shape:", X_train.shape)
    # print("Classes:", X_train.classes)

    #################

    dim_input = dataset.shape[1] - dim_output
    normalising_factor = 1.0
    hidden_layer_dims = [100, 150, 100]

    # Hyperparameters
    epochs = int(1e4)
    batch_size = 1000
    learning_rate = 1e-3

    # Flatten the dataset and normalise it
    # We flatten the dataset because we need to pass in a 1D array
    # not a 3D array (which is (X, Y, Channels))
    # X_train, y_train, X_test, y_test = preprocess_data(
    #     X_train, y_train, X_test, y_test, dim_input, normalising_factor
    # )

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

    print(f"{Fore.GREEN}Creating model for {dataset_name} dataset{Style.RESET_ALL}")
    # Instantiate the model and move it to the specified device
    sequential_model = model.create_sequential_model(
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

    optimiser = torch.optim.SGD(
        sequential_model.parameters(), lr=learning_rate)

    # --- Training Loop ---

    metrics = []
    for i in tqdm(range(int(epochs))):
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
            dim_output,
        )

    results.show_training_results(metrics)


def main():
    welcome()

    dataset_name = "Thorax"

    run_model(dataset_name)


if __name__ == "__main__":
    main()
