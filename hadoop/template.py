#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some description here...
"""

import sys
from operator import itemgetter
from itertools import groupby


def read_input(file):
    """Read input and split."""
    for line in file:
        yield line.rstrip().split('\t')


def main():
    data = read_input(sys.stdin)
    for key, kviter in groupby(data, itemgetter(0)):
        # some code here..

if __name__ == "__main__":
    main()
