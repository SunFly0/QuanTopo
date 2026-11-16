#!/usr/bin/env python3
"""Train QuanTopo Stage B1."""

import sys

from quantopo.cli.train import main

if __name__ == "__main__":
    main(["--stage", "b1", *sys.argv[1:]])
