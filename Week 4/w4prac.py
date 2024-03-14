"""
Week 4 Prac Main file
"""

from q1 import q1
from q2 import q2
import os
import sys


def main():
    current_folder = os.path.dirname(__file__)
    data_folder = os.path.join(current_folder, "data")

    questions = [str(i) for i in range(1, 2 + 1)]

    if len(sys.argv) > 1:
        if sys.argv[1] in questions:
            if sys.argv[1] == "1":
                q1()
            elif sys.argv[1] == "2":
                q2(data_folder)

    else:
        # Run all questions

        # Question 1
        q1()

        # Question 2
        q2(data_folder)


if __name__ == "__main__":
    main()
