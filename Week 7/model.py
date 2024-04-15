import torch
from typing import List


def create_sequential_model(dim_input: int, dim_output: int, hidden_layer_dims: List[int]) -> torch.nn.Sequential:
    """
    Create a sequential model

    Args:
        dim_input: Dimension of the input
        dim_output: Dimension of the output
        hidden_layers: List of hidden layers

    Returns:
        Sequential model
    """

    print("Input dimension: ", dim_input)

    hiddens = [dim_input, *hidden_layer_dims]

    torch_layers = []

    # Create a linear layer and feed it through the activation function
    for i in range(len(hiddens) - 1):
        # Create linear layer from i to i+1
        linear_layer = torch.nn.Linear(hiddens[i], hiddens[i+1])

        # Create activation function
        activation = torch.nn.GELU()

        # Add to the list
        torch_layers.append(linear_layer)
        torch_layers.append(activation)

    # Add the final output layer
    final_hidden_layer = hiddens[-1]
    output_layer = torch.nn.Linear(final_hidden_layer, dim_output)
    torch_layers.append(output_layer)

    # Turn the list into a sequential model
    sequential_model = torch.nn.Sequential(*torch_layers)

    return sequential_model
