import sys

from src.evaluate import evaluate_command, simulate_command
from src.train import train_command
from utils.generate_maps import generate_all_map


def main():
    command = sys.argv[1]
    print(f"Command: {command}")
    if command == "train":
        train_command()
    if command == "evaluate":
        evaluate_command()
    if command == "simulate":
        simulate_command()
    if command == "generate":
        [n_row, n_col] = sys.argv[2].split('X')
        print(f"Generating all possible maps of size {n_row}x{n_col}")
        generate_all_map(int(n_col), int(n_row))


main()
