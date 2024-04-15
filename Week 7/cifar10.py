"""
Week 7 Prac
CIFAR10 dataset classification
11/04/2024
"""

from classification import classification_model
from welcome import welcome


def main():
    welcome()

    dim_in = 32*32*3
    normalising_factor = 255.0
    dim_out = 10
    hidden_layer_dims = [100, 150, 100]

    # Hyperparameters
    epochs = int(1e4)
    batch_size = 20000
    learning_rate = 1e-3

    classification_model("CIFAR10", dim_in, dim_out,
                         hidden_layer_dims, normalising_factor,
                         epochs, batch_size, learning_rate)


if __name__ == "__main__":
    main()
