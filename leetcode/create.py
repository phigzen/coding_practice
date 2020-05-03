"""
python create.py "20. Valid Parentheses"
"""
import argparse
import os


def parse_args():
    description = "You should add those parameters: "
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(dest='names', metavar='filename', nargs='*')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args.names)
    for _dir in args.names:
        _dir = ' '.join(_dir.split())
        _dir = _dir.replace(' ', r'\ ')
        os.system(f"mkdir {_dir} && cd {_dir} && touch solution.py && cd ..")
