#!/usr/bin/env python3
from sys import argv

if __name__ == '__main__':
    filename = argv[1]
    squareSize = int(argv[2])
    f = open(filename, 'r')
    for line in f.readlines():
        l = line[:-1].split(',')
        i = 0
        while i+squareSize < len(l):
            print( ''.join(l[i:i+squareSize]))
            i += squareSize
        print("[" + ', '.join(l[i:]) + "]\n")






