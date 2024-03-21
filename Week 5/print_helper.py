from colorama import Fore, Style


def print_question_header(question_num: int) -> None:
    print(f"{Fore.LIGHTYELLOW_EX}Question {question_num}:{Style.RESET_ALL}")
