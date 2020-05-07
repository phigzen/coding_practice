"""
# 新版本：
python new.py 225. Implement Stack using Queues
# 旧版本：
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
    _dir = r'\ '.join(args.names)
    print(_dir)
    os.system(f"mkdir {_dir} && cd {_dir} && touch solution.py && echo {_dir} >> solution.py && cd ..")

    # for _dir in args.names:
    #     _dir = ' '.join(_dir.split())
    #     _dir = _dir.replace(' ', r'\ ')
    #     os.system(f"mkdir {_dir} && cd {_dir} && touch solution.py && cd ..")
