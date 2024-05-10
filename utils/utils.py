from typing import List
import numpy as np


def current_time_for_file():
    """
    This function returns the current time
    :return: current time
    """
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(" ", "_").replace(":", "_")


def print_map(map_array):
    for row in map_array:
        print(row)
    print("\n")


def simplify_key(key: str):
    """
    This function takes a string and returns a simplified version of the string
    :param key:
    :return:
    """
    key = key.lower()
    if "_" in key:
        temp = key.split("_")
        result = ""
        for i in temp:
            result += i[0]
        return result
    else:
        if len(key) > 1:
            return key[0:1]
        else:
            return key


def get_merge_dictionary(dict1: dict, dict2: dict):
    """
    This function returns a merged dictionary
    adding keys of dict1 and values of dict2
    :param dict1:
    :param dict2:
    :return: dictionary merged keys of dict1 and values of dict2
    """
    result = dict1
    for key in dict2:
        result[key] = dict2[key]
    return result


def is_valid(board: List[List[str]]) -> bool:
    n_row, n_col = len(board), len(board[0])

    start = (0, 0)
    for i in range(n_row):
        for j in range(n_col):
            if board[i][j] == b"S":
                start = (i, j)
                break
    frontier, discovered = [], set()
    frontier.append(start)
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= n_row or c_new < 0 or c_new >= n_col:
                    continue
                if board[r_new][c_new] == b"G":
                    return True
                if board[r_new][c_new] != b"H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(n_col, n_row, p):
    """
    This function generates a random map
    :param n_col: number of columns
    :param n_row: number of rows
    :param p: probability
    :return: random map
    """
    valid = False
    _map = []
    while not valid:
        _map = np.random.choice([b'F', b'H'], (n_row, n_col), p=[1 - p, p])
        start = np.random.choice(range(n_col))
        goal = np.random.choice(range(n_col))
        _map[0, start] = b'S'
        _map[n_row - 1, goal] = b'G'
        valid = is_valid(_map.tolist())

    return _map.tolist()
