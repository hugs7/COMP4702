def sample_mean(data: list[float]) -> float:
    """
    Calculate the sample mean of a list of numbers
    """

    return sum(data) / len(data)


def sample_standard_deviation(data: list[float]) -> float:
    """
    Calculate the sample standard deviation of a list of numbers
    """

    n = len(data)

    mean = sum(data) / n

    variance = sum((x - mean) ** 2 for x in data) / (n - 1)

    return variance**0.5
