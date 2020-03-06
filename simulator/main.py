import os


def run():
    for i in range(10):
        os.system("{ /usr/bin/time --format='\t%U\,\t%S' python interface.py; } 2>> ./log/log_10")


if __name__ == '__main__':
    run()
