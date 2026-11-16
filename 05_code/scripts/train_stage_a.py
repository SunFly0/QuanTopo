#!/usr/bin/env python3
"""Train QuanTopo Stage A."""

import sys

from quantopo.cli.train import main

if __name__ == "__main__":
    main(["--stage", "a", *sys.argv[1:]])
