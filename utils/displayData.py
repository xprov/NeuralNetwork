#!/usr/bin/env python3
from sys import argv, stdin

if __name__ == '__main__':
    if len(argv) == 2:
        print('aqui')
        iterator = stdin
        squareSize = int(argv[1])
    else:
        print('aqua')
        filename = argv[1]
        squareSize = int(argv[2])
        if filename == "-" :
            iterator = stdin
        else:
            iterator = open(filename, 'r').readlines()
    for line in iterator:
        l = line[:-1].split(',')
        i = 0
        while i+squareSize < len(l):
            print( ''.join(l[i:i+squareSize]))
            i += squareSize
        print("[" + ', '.join(l[i:]) + "]\n")






