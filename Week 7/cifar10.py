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
    hidden_layer_dims = [100, 100]
    classification_model("CIFAR10", dim_in, dim_out,
                         hidden_layer_dims, normalising_factor)


if __name__ == "__main__":
    main()
