#!/usr/bin/env python3

from sys import argv
from itertools import product

def genData(n):
    lines = []
    for i in product([0,1], repeat=n):
        s = ','.join(map(str, i))
        lines.append(s + "," + s)
    return lines

if __name__ == '__main__':
    n = int(argv[1]) # nb input/output nodes
    m = int(argv[2]) # nb data lines
    lines = genData(n)
    nbData = 2**n
    for i in range(m):
        print(lines[i%nbData])

