#!/usr/bin/env python3

from random import randint, random, shuffle
from sys import argv
from math import sqrt, pi, atan, floor, ceil

def singleOne(nBits, nToGen, goodRatio):
    l = []
    for i in range(nToGen):
        if random() < goodRatio:
            tmp = [0]*nBits
            tmp[randint(0,nBits-1)] = 1
            tmp.append(1)
        else:
            tmp = [0]*nBits
            error = randint(1,nBits)
            if error > 1 :
                for i in range(error):
                    done = False
                    while not done:
                        pos = randint(0,nBits-1)
                        if tmp[pos] == 0:
                            tmp[pos] = 1
                            done = True
            tmp.append(0)
        l.append(','.join(map(lambda x : str(x), tmp)))
    return l

def centerOfSquare(squareSize, centerSize, nToGen, goodRatio):
    """
    +------------------------------+
    |Here should be only zeros     |
    |                              |
    |                              |
    |                              |
    |                              |
    |         +----------+         |
    |         |All ones  |         |
    |         |should    |         |
    |         |be in     |         |
    |         |here      |         |
    |         |          |         |
    |         +----------+         |
    |                              |
    |                              |
    |                              |
    |                              |
    +------------------------------+
    """
    theSquares = []
    center = []
    boundary = []
    offset = squareSize//2 - centerSize//2
    inCenter = lambda x : offset <= x < offset+centerSize
    for l in range(squareSize):
        for c in range(squareSize):
            if inCenter(l) and inCenter(c):
                center.append(l*squareSize + c)
            else:
                boundary.append(l*squareSize + c)
    for _ in range(nToGen):
        square = [0]*(squareSize*squareSize)
        if random() < goodRatio:
            nbInCenter = max(len(center)//2, randint(1,len(center)))
            nbInBoundary = 0
            square.append(1)
        else :
            nbInCenter = min(len(center)//2, randint(1,len(center)))
            nbInBoundary = randint(1, len(boundary)-1)
            square.append(0)
        shuffle(center)
        for i in center[:nbInCenter]:
            square[i] = 1
        for i in boundary[:nbInBoundary]:
            square[i] = 1
        theSquares.append(square)
    return theSquares



class Matrix :
    def __init__(self, n, m):
        self._n = n
        self._m = m
        self._data = [0]*(n*m)

    def nRows(self):
        return self._n

    def nCols(self):
        return self._m

    def get(self, i, j):
        return self._data[i*self.nCols() + j]

    def normalize(self):
        """
        Divides all entries of the matrix by their sum so that the
        resulting matrix has sum equal to 1.
        """
        s = sum(self._data)
        if s > 0:
            self._data = [i/s for i in self._data]

    def set(self, i, j, value):
        self._data[i*self.nCols() + j] = value

    def getRandomPos(self):
        return (randint(0, self.nRows()-1), randint(0, self.nCols()-1))

    def drawRandomLine(self):
        # select start point 
        n = self.nRows()
        m = self.nCols()
        if random() < 0.5:
            start = (randint(n//2, n-1), 0)
        else :
            start = (n-1, randint(0, m//2))
        # select end point 
        if random() < 0.5:
            end = (0,randint(m//2, m-1))
        else :
            end = (randint(0, n//2), m-1)

        if start[1] == end[1] :
            for y in range(n):
                self.set(y, start[1], 1)
        elif start[0] == end[0] :
            for x in range(m):
                self.set(start[0], x, 1)
        else:
            y, x = start
            m = (end[0]-start[0])/(end[1]-start[1])
            while x <= end[1]:
                self.set(y, x, 1)
                if ( end[0] > start[0] ):
                    while y <= start[0]+m*(x-start[1]):
                        self.set(y, x, 1)
                        y += 1
                else:
                    while y >= start[0]+m*(x-start[1]):
                        self.set(y, x, 1)
                        y -= 1
                x += 1
        if random()<0.5:
            self.horizontalFlip()

    def horizontalFlip(self):
        m = self.nCols()
        for i in range(self.nRows()):
            for j in range(m//2):
                tmp = self.get(i, j)
                self.set(i, j, self.get(i, m-j-1))
                self.set(i, m-j-1, tmp)

    def drawRandomSmallSquare(self):
        p = randint(floor(2*sqrt(self.nRows())/3), ceil(3*sqrt(self.nRows())/2))
        q = randint(floor(2*sqrt(self.nCols())/3), ceil(3*sqrt(self.nCols())/2))
        a = randint(0, self.nRows()-p )
        b = randint(0, self.nCols()-q )
        startRow = a
        endRow = a+p
        startCol = b
        endCol = b+q
        for i in range(startRow, endRow):
            for j in range(startCol, endCol):
                self.set(i, j, 1)

    def drawRandomSquare(self):
        a = randint(0, self.nCols()-1 )
        b = randint(0, self.nCols()-1 )
        c = randint(0, self.nRows()-1 )
        d = randint(0, self.nRows()-1 )
        startCol = min(a,b)
        endCol = max(a,b)
        startLine = min(c,d)
        endLine = max(c,d)
        for i in range(startLine, endLine):
            for j in range(startCol, endCol):
                self.set(i, j, 1)

    def drawRandomShit(self):
        n = randint(floor(2*sqrt(len(self._data))/3), ceil(3*sqrt(len(self._data))/2))
        self._data = ([1]*n) + ([0]*(len(self._data)-n))
        shuffle(self._data)




    def __repr__(self):
        l = []
        for i in range(self.nRows()):
            l.append(self._data[i*self.nCols() : (i+1)*self.nCols()])
        return "\n".join(map(str, l))

    def linearOutput(self):
        return ','.join(map(str, self._data))


def genLines(squareSize, nToGen, goodRatio):
    for i in range(nToGen):
        m = Matrix( squareSize, squareSize )
        r = random()
        if r < 0.33:
            m.drawRandomLine()
            suffix = ",1,0,0"
        elif r < 0.66:
            #m.drawRandomSquare()
            m.drawRandomSmallSquare()
            suffix = ",0,1,0"
        else :
            m.drawRandomShit()
            suffix = ",0,0,1"
        #print(m)
        #m.normalize()
        print(str(m.linearOutput()) + suffix)















#if __name__ == '__main__':
#    lng = int(argv[1])
#    nb = int(argv[2])
#    ratio = float(argv[3])
#    l = singleOne(lng, nb, ratio)
#    print('\n'.join(l))


if __name__ == '__main__':
    size = int(argv[1])
    n = int(argv[2])
    genLines(size, n, 0.5)
    

    ###squareSize = int(argv[1])
    ###centerSize = int(argv[2])
    ###nToGen = int(argv[3])
    ###ratio = float(argv[4])
    ###l = centerOfSquare(squareSize, centerSize, nToGen, ratio)
    ####displaySquares(l)
    ###print('\n'.join(','.join(map(str, i)) for i in l))


