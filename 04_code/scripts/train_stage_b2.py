#!/usr/bin/env python3
"""Train QuanTopo Stage B2."""

import sys

from quantopo.cli.train import main

if __name__ == "__main__":
    main(["--stage", "b2", *sys.argv[1:]])
