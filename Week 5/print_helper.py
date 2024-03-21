from colorama import Fore, Style


def print_question_header(question_num: int) -> None:
    print(f"{Fore.LIGHTYELLOW_EX}Question {question_num}:{Style.RESET_ALL}")


def print_sample_mean(question_num: int, sample_mean: float) -> None:
    print(
        f"{Fore.LIGHTMAGENTA_EX}Question {question_num} sample mean: {Style.RESET_ALL}",
        sample_mean,
    )


def print_sample_standard_deviation(
    question_num: int, sample_standard_deviation: float
) -> None:
    print(
        f"{Fore.LIGHTMAGENTA_EX}Question {question_num} sample standard deviation: {Style.RESET_ALL}",
        sample_standard_deviation,
    )
