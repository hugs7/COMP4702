"""
Week 4 Prac Main file
"""

import matplotlib.pyplot as plt
import numpy as np


def q1():
    # Part A

    f = lambda x: x**3 + 1

    domain = np.linspace(-1, 1, 1000)

    # Plot the function
    plt.plot(domain, f(domain))
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    # Part B

    # Part C


def q2():
    pass


def main():
    print("Week 4 Prac Main file")

    # Question 1
    q1()


if __name__ == "__main__":
    main()
