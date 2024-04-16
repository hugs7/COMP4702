"""
Main Driver file for project
"""

from tqdm import tqdm
from classification import preprocess_data
from welcome import welcome
import train
import model
import torch
import os
from colorama import Fore, Style
import read_data


def run_model(dataset_name: str) -> None:
    dataset_mapping = {
        "Thorax": "83_Loeschcke_et_al_2000_Thorax_&_wing_traits_lab pops.csv",
        "Wing_traits": "84_Loeschcke_et_al_2000_Wing_traits_&_asymmetry_lab pops.csv",
        "Wing_asymmetry": "85_Loeschcke_et_al_2000_Wing_asymmetry_lab_pops.csv",
    }

    if dataset_name not in dataset_mapping:
        raise ValueError(f"{Fore.RED}Dataset {dataset_name} not found{Style.RESET_ALL}")

    dataset_file_name = dataset_mapping[dataset_name]

    # --- Dataset ---

    folder_of_script = os.path.dirname(__file__)
    data_folder = os.path.join(folder_of_script, "data")

    # Create data folder if it does not exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    model_data_path = os.path.join(data_folder, dataset_file_name)

    # Read the dataset
    dataset = read_data.read_excel_file(model_data_path)
    print(dataset)
    exit(0)

    train_data = mappped_dataset(
        model_data_path, download=download, train=True, transform=True
    )
    validation_data = mappped_dataset(
        model_data_path, download=download, train=False, transform=True
    )

    print("Training data stats")
    print("Shape:", train_data.data.shape)
    print("Classes:", train_data.classes)

    #################

    dim_out = 2  # Things we are predicting
    dim_in = dataset.shape[1] - dim_out
    normalising_factor = 255.0
    hidden_layer_dims = [100, 150, 100]

    # Hyperparameters
    epochs = int(1e4)
    batch_size = 20000
    learning_rate = 1e-3

    # Flatten the dataset and normalise it
    # We flatten the dataset because we need to pass in a 1D array
    # not a 3D array (which is (X, Y, Channels))
    train_data, train_labels, validation_data, validation_labels = preprocess_data(
        train_data, validation_data, dim_input, normalising_factor
    )

    # Move data to device
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    validation_data = validation_data.to(device)
    validation_labels = validation_labels.to(device)

    print("Training data shape:", train_data.shape)
    print("Validation data shape:", validation_data.shape)

    # ----- Device ------
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{Fore.GREEN}Creating model for {dataset_name} dataset{Style.RESET_ALL}")
    # Instantiate the model and move it to the specified device
    sequential_model = model.create_sequential_model(
        dim_input, dim_output, hidden_layer_dims
    ).to(device)

    print(sequential_model)

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
    for i in tqdm(range(int(optimisation_steps))):
        metrics = train.nn_train(
            i,
            train_data,
            train_labels,
            batch_size,
            sequential_model,
            criterion,
            optimiser,
            optimisation_steps,
            metrics,
        )

    show_training_results(metrics)


def main():
    welcome()

    dataset_name = "Thorax"

    run_model(dataset_name)


if __name__ == "__main__":
    main()
