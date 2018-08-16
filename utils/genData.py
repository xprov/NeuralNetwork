#!/usr/bin/env python3

from random import randint, random
from sys import argv, stderr
from math import sqrt, pi, floor, ceil, acos, pi



# Brezenham
#
# Taken from https://github.com/encukou/bresenham
#
# Copyright © 2016 Petr Viktorin
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# 
# Implementation of Bresenham's line drawing algorithm
# See en.wikipedia.org/wiki/Bresenham's_line_algorithm
#
def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

def distance(A, B):
    return sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

def angle(A, B, C):
    """
    ^
    |
    |                        
    |              B---------------C
    |             / 
    |            /  Angle here
    |           /   (in degrees)
    |          /    (unsigned)         
    |         /
    |        /
    |       A 
    |
    +---------------------------------------------------->
    """
    x1 = A[1]-B[1]
    y1 = A[0]-B[0]
    x2 = C[1]-B[1]
    y2 = C[0]-B[0]
    lng1 = distance(A, B)
    lng2 = distance(B, C)
    dot_p = x1*x2 + y1*y2
    value = dot_p / (lng1*lng2) 
    # if rounding error get the value outside of [-1,1], then correct it
    value = max( value, -1.0 )
    value = min( value, 1.0 )
    return 180/pi * acos(value)


class Matrix :
    def __init__(self, nRows, nCols):
        self._n = nRows
        self._m = nCols
        self._data = [0]*(self._n * self._m)

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

    def randomPoint(self):
        return (randint(0, self.nRows()-1), randint(0, self.nCols()-1))

    def randomPointOnBoundary(self):
        side = randint(0,3)
        rowMax = self.nRows()-1
        colMax = self.nCols()-1
        randRow = lambda : randint(0, rowMax)
        randCol = lambda : randint(0, colMax)
        if side == 0 : 
            # top side
            return (0, randCol())
        if side == 1:
            # left side
            return (randRow(), 0)
        if side == 2:
            # bottom side
            return (rowMax, randCol())
        if side == 3:
            # right side
            return (randRow(), colMax)

    def drawRandomLine(self):
        lngMin = min(self.nCols(), self.nRows()) // 2
        ok = False
        while not ok:
            r1,c1 = self.randomPointOnBoundary()
            r2,c2 = self.randomPointOnBoundary()
            ok = distance((r1,c1), (r2,c2)) > lngMin
        for r,c in bresenham(r1, c1, r2, c2):
            self.set( r, c, 1 )

    def drawTriangle(self, A, B, C):
        # Frist, compute the left and right bondaries of the triangle on each row
        leftBoundary = [self.nCols()]*self.nRows()
        rightBoundary = [-1]*self.nRows() 

        # From A to B
        for (r,c) in bresenham(A[0], A[1], B[0], B[1]):
            leftBoundary[r] = min(leftBoundary[r], c)
            rightBoundary[r] = max(rightBoundary[r], c)

        # From A to C
        for (r,c) in bresenham(A[0], A[1], C[0], C[1]):
            leftBoundary[r] = min(leftBoundary[r], c)
            rightBoundary[r] = max(rightBoundary[r], c)

        # From B to C
        for (r,c) in bresenham(B[0], B[1], C[0], C[1]):
            leftBoundary[r] = min(leftBoundary[r], c)
            rightBoundary[r] = max(rightBoundary[r], c)

        # Second, fill da triangle 
        for r in range(self.nRows()):
            for c in range(leftBoundary[r], rightBoundary[r]+1):
                self.set(r, c, 1)

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
        ok = False
        minWidth  = min(2,  self.nCols() // 2)
        maxWidth  = min(15, self.nCols() // 2)
        minHeight = min(2,  self.nRows() // 2)
        maxHeight = min(15, self.nRows() // 2)
        while not ok:
            a = randint(0, self.nCols()-1 )
            b = randint(0, self.nCols()-1 )
            c = randint(0, self.nRows()-1 )
            d = randint(0, self.nRows()-1 )
            startCol  = min(a,b)
            endCol    = max(a,b)
            startRow  = min(c,d)
            endRow    = max(c,d)
            ok = (endCol-startCol >= minWidth)\
                 and (endCol-startCol <= maxWidth)\
                 and (endRow-startRow >= minHeight)\
                 and (endRow-startRow <= maxHeight)
        for i in range(startRow, endRow):
            for j in range(startCol, endCol):
                self.set(i, j, 1)

    def drawRandomPoints(self):
        """
        Draw n random points.
        """
        n = len(self._data)
        #minNumPoints = min( self.nCols(), self.nRows() )
        #maxNumPoints = n - max(self.nCols(), self.nRows())
        minNumPoints = 0
        if random() < 0.75 :
            maxNumPoints = floor(sqrt(self.nRows() * self.nCols()))
        else:
            maxNumPoints = self.nRows() * self.nCols() - 1
        nbPoints = randint(minNumPoints, maxNumPoints)
        for i in range(nbPoints):
            A = self.randomPoint()
            self.set(A[0], A[1], 1)

    def drawRandomTriangle(self):
        ok = False
        minSideLength = min(5,  self.nCols()//2, self.nRows()//2)
        maxSideLength = min(15, self.nCols()//2, self.nRows()//2)
        minAngle = 30.0
        while not ok :
            A = self.randomPoint()
            B = self.randomPoint()
            C = self.randomPoint()
            ok = (\
                  distance(A, B) >= minSideLength \
                  and distance(A, C) >= minSideLength\
                  and distance(B, C) >= minSideLength\
                  and distance(A, B) <= maxSideLength \
                  and distance(A, C) <= maxSideLength\
                  and distance(B, C) <= maxSideLength\
                  and angle(A, B, C) >= minAngle\
                  and angle(A, C, B) >= minAngle\
                  and angle(B, A, C) >= minAngle\
                 )
        self.drawTriangle(A, B, C)






    def __repr__(self):
        l = []
        for i in range(self.nRows()):
            l.append(self._data[i*self.nCols() : (i+1)*self.nCols()])
        return "\n".join(map(str, l))

    def serialize(self):
        return ','.join(map(str, self._data))


def genData(width, height, n):
    for i in range(n):
        m = Matrix(height, width)
        r = random()
        if r < 0.25:
            m.drawRandomLine()
            suffix = ",1,0,0"
        elif r < 0.5:
            m.drawRandomSquare()
            suffix = ",0,1,0"
        elif r < 0.75:
            m.drawRandomTriangle()
            suffix = ",0,0,1"
        else:
            m.drawRandomPoints()
            suffix = ",0,0,0"
        #print(m)
        #m.normalize()
        print(str(m.serialize()) + suffix)


if __name__ == '__main__':
    size = int(argv[1])
    n = int(argv[2])
    genData(size, size, n)
    


