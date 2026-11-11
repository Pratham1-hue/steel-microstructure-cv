import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen
import measure


def main():
    gen.main()
    measure.main()


if __name__ == "__main__":
    main()
